import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from datasets.io import IO
import numpy as np
import random

def runGan_net(args,config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    Gnet_model=builder.model_builder(config.GnetModel)
    Dnet_model=builder.model_builder(config.DnetModel)
    Gnet_model = nn.DataParallel(Gnet_model).cuda()
    Dnet_model = nn.DataParallel(Dnet_model).cuda()
    if args.use_gpu:
        Gnet_model.to(args.local_rank)
        Dnet_model.to(args.local_rank)

    # manualSeed = random.randint(1, 10000)
    # print("Random Seed: ", manualSeed)
    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)
    # torch.cuda.manual_seed_all(manualSeed)

    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # parameter setting
    start_epoch = 0
    best_pre_gt = None
    metrics_pre_gt = None
    best_gt_pre = None
    metrics_gt_pre = None
    best_pre_gt_all = None
    metrics_pre_gt_all = None
    best_gt_pre_all = None
    metrics_gt_pre_all = None

    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    criterion_PointLoss = ChamferDistanceL2()

    optimizerG, schedulerG = builder.build_opti_sche(Gnet_model, config)
    optimizerD, schedulerD = builder.build_opti_sche(Dnet_model, config)

    real_label = 1
    fake_label = 0

    ###########################
    #  G-NET and T-NET
    ##########################

    for epoch in range(start_epoch,config.max_epoch):
        if epoch<30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2

        epoch_start_time = time.time()
        batch_start_time = time.time()
        #batch_time = AverageMeter()
        #data_time = AverageMeter()
        losses = AverageMeter(['CD_LOSS', 'Gt_Pre', 'Pre_Gt', 'errG_l2', 'errG_D', 'errD', 'errG'])
        n_batches = len(train_dataloader)
        label = torch.FloatTensor(n_batches)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            #data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                # gt = data[1].cuda()
                # if config.dataset.train._base_.CARS:
                #     if idx == 0:
                #         print_log('padding while KITTI training', logger=logger)
                #     partial = misc.random_dropping(partial, epoch)  # specially for KITTI finetune
            elif dataset_name == 'ShapeNet':
                #real_point, target = data
                real_point = data
                real_point = real_point.cuda()
                fixed_points=[torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])] #设置5个点的视角
                input_cropped, real_center,_ = misc.seprate_point_cloud(real_point, npoints, config.GnetModel.crop_point_num, fixed_points=fixed_points,padding_zeros=True)
                real_center = real_center.cuda()
                input_cropped = input_cropped.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if isinstance(config.GnetModel.point_scales_list, list):
                input_cropped=misc.gather_multiResolution_point_cloud(input_cropped,config.GnetModel.point_scales_list)

            label.resize_([real_point.size()[0], 1]).fill_(real_label)
            label=label.cuda()

            Gnet_model = Gnet_model.train()
            Dnet_model = Dnet_model.train()
            ############################
            # (2) Update D network
            ###########################
            Dnet_model.zero_grad()
            real_center = torch.unsqueeze(real_center,1)
            output = Dnet_model(real_center)
            errD_real = criterion(output,label)
            errD_real.backward()

            fake_center1,fake_center2,fake = Gnet_model(input_cropped)
            fake = torch.unsqueeze(fake,1)
            label.data.fill_(fake_label)
            output = Dnet_model(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            Gnet_model.zero_grad()
            label.data.fill_(real_label)
            output = Dnet_model(fake)
            errG_D = criterion(output, label)
            errG_l2 = 0
            real_center_key1=misc.fps(torch.squeeze(real_center, 1),64)
            real_center_key2=misc.fps(torch.squeeze(real_center, 1),256)

            CD_LOSS, Pre_Gt, Gt_Pre = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
            CD_fc1_rc1, _, _ = criterion_PointLoss(fake_center1,real_center_key1)
            CD_fc2_rc2, _, _ = criterion_PointLoss(fake_center1, real_center_key2)
            errG_l2 = CD_LOSS+alpha1*CD_fc1_rc1+alpha2*CD_fc2_rc2

            errG = (1-config.wtl2) * errG_D + config.wtl2 * errG_l2
            errG.backward()
            optimizerG.step()

            losses.update([CD_LOSS.item(), Pre_Gt.item(), Gt_Pre.item(), errG_l2.item(), errG_D.item(), errD.item(), errG.item()])
            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/CD_Loss', CD_LOSS.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Pre_Gt', Pre_Gt.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Gt_Pre', Gt_Pre.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/errG_l2', errG_l2.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/errG_D', errG_D.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/errD', errD.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/errG', errG.item(), n_itr)

            # batch_time.update(time.time() - batch_start_time)
            # batch_start_time = time.time()
            # if idx % 20 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lrGnet = %.6f lrDnet = %.6f'  %
            #               (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                ['%.4f' % l for l in losses.val()], optimizerG.param_groups[0]['lr'], optimizerD.param_groups[0]['lr']), logger=logger)
        if isinstance(schedulerG, list):
            for item in schedulerG:
                item.step(epoch)
        else:
            schedulerG.step(epoch)

        if isinstance(schedulerD, list):
            for item in schedulerD:
                item.step(epoch)
        else:
            schedulerD.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/CD_loss', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Batch/Pre_Gt', losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Batch/Gt_Pre', losses.avg(2), epoch)
            train_writer.add_scalar('Loss/Epoch/errG_l2', losses.avg(3), epoch)
            train_writer.add_scalar('Loss/Epoch/errG_D', losses.avg(4), epoch)
            train_writer.add_scalar('Loss/Epoch/errD', losses.avg(5), epoch)
            train_writer.add_scalar('Loss/Epoch/errG', losses.avg(6), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics_pre_gt, metrics_gt_pre, metrics_pre_gt_all, metrics_gt_pre_all = validateGan(Gnet_model, Dnet_model, test_dataloader, epoch, None, criterion_PointLoss, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if metrics_pre_gt.better_than(best_pre_gt):
                best_pre_gt = metrics_pre_gt
                # builder.save_checkpoint(Gnet_model, optimizerG, epoch, metrics_pre_gt, best_pre_gt, f'best-pre-gt-G', args, logger=logger)
                # builder.save_checkpoint(Dnet_model, optimizerD, epoch, metrics_pre_gt, best_pre_gt, f'best-pre-gt-D', args, logger=logger)
                print_log('[Save checkpoints] best-pre-gt EPOCH: %d' %  epoch, logger=logger)
            if metrics_gt_pre.better_than(best_gt_pre):
                best_gt_pre = metrics_gt_pre
                # builder.save_checkpoint(Gnet_model, optimizerG, epoch, metrics_gt_pre, best_gt_pre, f'best-gt-pre-G', args, logger=logger)
                # builder.save_checkpoint(Dnet_model, optimizerD, epoch, metrics_gt_pre, best_gt_pre, f'best-gt-pre-D', args, logger=logger)
                print_log('[Save checkpoints] best-gt-pre EPOCH: %d' % epoch, logger=logger)
            if metrics_pre_gt_all.better_than(best_pre_gt_all):
                best_pre_gt_all = metrics_pre_gt_all
                # builder.save_checkpoint(Gnet_model, optimizerG, epoch, metrics_pre_gt_all, best_pre_gt_all, f'best-pre-gt-all-G', args, logger=logger)
                # builder.save_checkpoint(Dnet_model, optimizerD, epoch, metrics_pre_gt_all, best_pre_gt_all, f'best-pre-gt-all-D', args, logger=logger)
                print_log('[Save checkpoints] best-pre-gt-all EPOCH: %d' % epoch, logger=logger)
            if metrics_gt_pre_all.better_than(best_gt_pre_all):
                best_gt_pre_all = metrics_gt_pre_all
                # builder.save_checkpoint(Gnet_model, optimizerG, epoch, metrics_gt_pre_all, best_gt_pre_all, f'best-gt-pre-all-G', args, logger=logger)
                # builder.save_checkpoint(Dnet_model, optimizerD, epoch, metrics_gt_pre_all, best_gt_pre_all, f'best-gt-pre-all-D', args, logger=logger)
                print_log('[Save checkpoints] best-gt-pre-all EPOCH: %d' % epoch, logger=logger)
        # builder.save_checkpoint(Gnet_model, optimizerG, epoch, metrics, best_metrics, 'ckpt-last-G', args, logger=logger)
        # builder.save_checkpoint(Dnet_model, optimizerD, epoch, metrics, best_metrics, 'ckpt-last-D', args, logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(Gnet_model, optimizerG, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}-G', args, logger=logger)
        #     builder.save_checkpoint(Dnet_model, optimizerD, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}-D', args, logger=logger)


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)


    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                #gt:[96,8192,3],npoints:8192,[2048,6144]
                #partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                input_cropped, partial,_ = misc.seprate_point_cloud(gt, npoints, int(npoints * 1 / 4),  fixed_points=None,padding_zeros=True)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1

            ret = base_model(input_cropped)
            
            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)
         
            _loss = sparse_loss + dense_loss 
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0  and epoch > 500:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    train_writer.close()
    val_writer.close()

def validateGan(Gnet_model, Dnet_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    Gnet_model.eval()
    Dnet_model.eval()
    test_metrics=AverageMeter(Metrics.names())
    test_metrics_all=AverageMeter(Metrics.names())
    test_losses_table2 = AverageMeter(['CD_LOSS', 'Pre_Gt', 'Gt_Pre'])
    test_losses_table1 = AverageMeter(['Pre_GT_All', 'Gt_Pre_All'])
    category_metrics = dict()
    category_metrics_all = dict()
    n_samples = len(test_dataloader)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data
                gt = gt.cuda()
                fixed_points = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                                torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
                input_ori, partial, ids = misc.seprate_point_cloud(gt, npoints, config.GnetModel.crop_point_num,
                                                          fixed_points=fixed_points, padding_zeros=True)
                partial = partial.cuda()
                input_ori = input_ori.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if isinstance(config.GnetModel.point_scales_list, list):
                input=misc.gather_multiResolution_point_cloud(input_ori, config.GnetModel.point_scales_list)

            Gnet_model.eval()
            fake_center1, fake_center2, fake = Gnet_model(input)

            real_input=input_ori.clone().detach()
            real_input[0,ids[:512]]=partial
            fake_input=input_ori.clone().detach()
            fake_input[0,ids[:512]]=fake
            _, Pre_GT_ALL, GT_Pre_All = ChamferDisL2(torch.squeeze(fake_input,1),torch.squeeze(real_input,1))
            test_losses_table1.update([Pre_GT_ALL, GT_Pre_All])
            CD_LOSS, Pre_Gt, Gt_Pre = ChamferDisL2(torch.squeeze(fake, 1), torch.squeeze(partial, 1))
            test_losses_table2.update([CD_LOSS.item(), Pre_Gt.item(), Gt_Pre.item()])
            #_metrics = Metrics.get(fake, gt)
            _metrics = Metrics.get(torch.squeeze(fake, 1), torch.squeeze(partial, 1))
            _metrics_all = Metrics.get(torch.squeeze(fake_input,1),torch.squeeze(real_input,1))

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if taxonomy_id not in category_metrics_all:  #table1
                category_metrics_all[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics_all[taxonomy_id].update(_metrics_all)

            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = input[0].squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input' % idx, input_pc, epoch, dataformats='HWC')
            #
            #     sparse = fake.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/fake' % idx, sparse_img, epoch, dataformats='HWC')
            #
            #     gt_ptcloud = gt.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
            # if (idx + 1) % 200 == 0:
            #     # print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
            #     #           (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()],
            #     #            ['%.4f' % m for m in _metrics]), logger=logger)
            #     print_log('Test[%d/%d] Taxonomy = %s Sample = %s  Metrics = %s' %
            #               (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % m for m in _metrics]), logger=logger)
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        for _, v in category_metrics_all.items():
            test_metrics_all.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.5f' % m for m in test_metrics.avg()]), logger=logger)
        print_log('[Validation] EPOCH: %d  Metrics_all = %s' % (epoch, ['%.5f' % m for m in test_metrics_all.avg()]), logger=logger)
        print_log('[table2] EPOCH: %d  CD_Loss Pre_Gt Gt_Pre= %s' % (epoch, ['%.5f' % m for m in test_losses_table2.avg()]), logger=logger)
        print_log('[table1] EPOCH: %d  Pre_Gt_All Gt_Pre_ALL= %s' % (epoch, ['%.5f' % m for m in test_losses_table1.avg()]), logger=logger)
        test_metrics = get_test_metrics(test_metrics,logger,category_metrics)
        test_metrics_all = get_test_metrics(test_metrics_all, logger, category_metrics_all)
        # print_log('Test[%d/%d]  Losses = %s Metrics = %s' %
        #           (idx + 1, n_samples, ['%.4f' % l for l in test_losses.val()],
        #            ['%.4f' % m for m in _metrics]), logger=logger)

        # Add testing results to TensorBoard
        if val_writer is not None:
            #val_writer.add_scalar('Loss/Batch/CD_Loss', CD_LOSS.item(), epoch)
            for i, metric in enumerate(test_metrics.items):
                val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)
        return Metrics('Pre_Gt', test_metrics.avg()), \
               Metrics('Gt_Pre', test_metrics.avg()), \
               Metrics('Pre_Gt',test_metrics_all.avg()),\
               Metrics('Gt_Pre',test_metrics_all.avg())

def get_test_metrics(test_metrics=None, logger=None, category_metrics=None):
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.4f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.5f \t' % value
    print_log(msg, logger=logger)

    return test_metrics

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                #input_cropped, partial,_ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None) PFNet DFNet
                input_cropped, partial, _ = misc.seprate_point_cloud(gt, npoints,  int(npoints * 1 / 4), fixed_points=None,padding_zeros=True)

                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(input_cropped)
            coarse_points = ret[0]
            dense_points = ret[1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt)
            # _metrics = [dist_utils.reduce_tensor(item, args) for item in _metrics]

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = partial.squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')
            #
            #     sparse = coarse_points.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')
            #
            #     dense = dense_points.squeeze().cpu().numpy()
            #     dense_img = misc.get_ptcloud_img(dense)
            #     val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
            #
            #     gt_ptcloud = gt.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            # if (idx+1) % 200 == 0:
            #     print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
            #                 (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()],
            #                 ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    test_metrics = get_test_metrics(test_metrics, logger, category_metrics)
    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)
    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median': 1/2,
    'hard': 3/4
}



def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())  #['F-Score','CDL1','CDL2']
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[1]

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points ,gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()#[1,8192,3]

                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]  #[1,192,3]
                    dense_points = ret[1] #[1,8192,3]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)

                    # test_metrics.update(_metrics)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return

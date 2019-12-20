import os
import argparse
import torch
import numpy as np
import scipy.misc
import torch.nn.functional as F
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import vis_res
import warnings
warnings.filterwarnings("ignore")


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=True,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    n_class = config['arch']['args']['n_class']
    dataset = config['name'].lower()
    with torch.no_grad():
        for idx, (data, gt, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output_cams = model.forward_cam(data)
            output = model(data)

            # Save sample images(CAMs)
            cam = F.upsample(output_cams, data.shape[2:], mode='bilinear', align_corners=False)[0]
            cam = cam.cpu().numpy() * target.cpu().clone().view(n_class, 1, 1).numpy()
            norm_cam = cam / (np.max(cam, (1, 2), keepdims=True) + 1e-5)
            cam_dict = {}
            for i in range(n_class):
                if target[0][i] > 1e-5:
                    cam_dict[i] = norm_cam[i]
            if cam_dict == {}: continue
            # save image
            save_path = os.path.join('saved', dataset, 'images', str(idx) + '.png')
            scipy.misc.imsave(save_path, data[0].cpu().permute(1, 2, 0).numpy())
            # save image
            save_path = os.path.join('saved', dataset, 'gt', str(idx) + '.npy')
            np.save(save_path, gt[0])
            # out cam
            save_path = os.path.join('saved', dataset, 'cams', str(idx) + '.npy')
            np.save(save_path, cam_dict)
            # out cam pred
            bg_score = [np.ones_like(norm_cam[0]) * 0.2]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            save_path = os.path.join('saved', dataset, 'cams_pred', str(idx) + '.png')
            vis_res(data[0][0].cpu(), gt[0].cpu().numpy().astype(np.uint8), pred.astype(np.uint8), save_path)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size


    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Weakly supervised learning for Medical Image')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

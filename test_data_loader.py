import sys
sys.path.insert(0, '.')

from data_loader import get_data_loader, get_dataset_config

print('Testing data_loader module...')

print('\n1. Testing dataset configs:')
for dataset in ['mnist', 'celeba', 'cifar10']:
    config = get_dataset_config(dataset)
    print('  %s: channels=%d, img_size=%d, is_gray=%s' % (
        dataset, config['channels'], config['img_size'], config['is_gray']))

print('\n2. Testing CIFAR-10 data loader (will download if not exists):')
try:
    loader = get_data_loader(
        dataset_name='cifar10',
        batch_size=4,
        data_dir='./data',
        img_size=64,
        num_workers=0
    )
    print('  Data loader created successfully!')
    print('  Number of batches: %d' % len(loader))
    
    for batch_idx, (data, target) in enumerate(loader):
        print('  Batch %d: data shape=%s, target shape=%s' % (
            batch_idx, data.shape, target.shape))
        print('  Data range: [%.3f, %.3f]' % (data.min(), data.max()))
        if batch_idx >= 2:
            break
    
    print('\nTest passed! CIFAR-10 dataset is ready for training.')
    
except Exception as e:
    print('  Error: %s' % str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\nAll tests passed!')

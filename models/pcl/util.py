def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images,is_eval=True) 
            features[index] = feat
    dist.barrier()        
    dist.all_reduce(features, op=dist.ReduceOp.SUM)     
    return features.cpu()

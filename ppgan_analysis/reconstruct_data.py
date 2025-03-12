from invertinggradients import inversefed
import torch 
import torchvision

def reconstruct_gradient(model_estimate_t1, model_estimate_t2, lr):
    # reconstruct the gradient list from model input and output at time t1 and t2
    # as we use gradient descent, it should be (model_t2-model_t1)/learning_rate
    input_gradient = []
    for p, q in zip(model_estimate_t1.parameters(), model_estimate_t2.parameters()):
        gradient = (q-p)/lr
        input_gradient.append(gradient)
    return input_gradient

def reconstruct_data(input_gradient, full_model_estimate_t1, loss_fn, config):
    setup = inversefed.utils.system_startup(config)
    dm = torch.as_tensor(getattr(inversefed.consts, f"{config.dataset}_mean"), **setup)[:,None,None]
    ds = torch.as_tensor(getattr(inversefed.consts, f"{config.dataset}_std"), **setup)[:,None,None]
    if config.dataset == 'mnist':
        img_shape = (1,28,28)
    elif config.dataset == 'cifar10':
        img_shape = (3,64,64)
    elif config.dataset == "celeba":
        img_shape = (3,64,64)
    invert_config = dict(
        signed = False,
        boxed=True,
        cost_fn='sim',
        indices = 'def',
        weights='equal',
        lr=0.01,
        optim='adam',
        restarts=1,
        max_iterations=18_000,
        total_variation=1e-1,
        init='randn',
        filter='none',
        lr_decay=True,
        scoring_choice='loss'
    )
    # print(next(full_model_estimate_t1.parameters()).device)
    rec_machine = inversefed.GradientReconstructor(full_model_estimate_t1, loss_fn, (dm,ds), invert_config, num_images=config.batch_size_real)
    output, stats = rec_machine.reconstruct(input_gradient, torch.zeros(config.batch_size_real).to("cuda"), img_shape=img_shape, dryrun=False)
    output_denormalized = torch.clamp(output*ds + dm, 0, 1)
    return output
    

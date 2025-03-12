import torch
def recon_model(input, output, iteration, schedule_step, criterion, model_generator, *args):
    # print(args)
    def get_closure(output,estimate,criterion,verbose):
        loss = criterion(output,estimate)
        loss.backward()
        def closure():
            if verbose:
                print(loss.item())
            return loss
        return closure
    model = model_generator.get_model(*args).secured_layers.to("cuda")
    # optimizer = torch.optim.LBFGS(model.parameters(),lr=0.03)#,weight_decay=0.01)
    # optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)
    # loss = 0
    for i in range(iteration):
        estimate = model(input)
        optimizer.zero_grad()
        # loss = criterion(output,estimate)
        # loss.backward()
        verbose=False
        if i == iteration-1:
            # print the loss of the last iteration
            verbose=True
        closure = get_closure(output,estimate,criterion,verbose)
        optimizer.step(closure)
        # if i % schedule_step == 0 and i!= 0:
        #     scheduler.step()
        
    return model

# if __name__=="__"

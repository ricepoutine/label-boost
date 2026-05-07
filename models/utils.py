def process_state_dict(pt_state_dict: dict, default_name = 'resnet'):
    # configures resnet18 weights to work with custom resnet, without classifier weights
    custom_state_dict = {}
    name_skip_index = 6 if default_name == 'resnet' else 6 # what is it for vitmae?? 
    for k, v in pt_state_dict:
        if default_name in k: # expected for every key... really wish the names can be set...
            custom_state_dict['0'+k[6:]] = v
    return custom_state_dict


def config(args):
    if args.dataset == 'ids18':
        # ===================================
        args.expriment_type = 'main'    # main, metasize, unannotated
        args.warmup_dir_path = './results/models/warmup/IDS18'
        args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_70r_21ep.pth'
        args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'
        # args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_balance_001_85.pth'

        # change in different experiments
        if args.expriment_type == 'main':
            args.corruption_type = 'sy'  # sy, asy,
            args.corruption_prob = 0.7

            args.meta_type = 'random'
            args.meta_size_prob = 0.01
            args.dc_expend_ratio = 0.2
            args.ceweight = [0.5, 3.6, 1.1, 3.5, 0.5, 2]

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_70r_21ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'

        elif args.expriment_type == 'metasize':
            args.corruption_type = 'asy'  # asy, sy
            args.corruption_prob = 0.5

            args.meta_type = 'balance'
            args.meta_size_prob = 500    #500, 100, 50
            args.dc_expend_ratio = 0.1
            args.ceweight = [1, 1, 1, 1, 1, 1]
            # args.ceweight = [0.5, 3.6, 1.1, 3.5, 0.5, 2]

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_50r_33ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_balance_500_85.pth'

        elif args.expriment_type == 'unannotated':
            args.corruption_type = 'ask'  # ask
            args.corruption_prob = 0.3  # 0.3  0.1

            args.meta_type = 'balance'
            args.meta_size_prob = 500
            args.dc_expend_ratio = 0.2

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_ask_3r_14ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_balance_500_85.pth'

            if args.corruption_prob == 0.1:
                args.ceweight = [1, 3.6, 1, 1, 1, 1]
            elif args.corruption_prob == 0.3:
                args.ceweight = [1, 1, 1, 3.5, 1, 1]
        # ====================================

        args.data_path = './data'
        args.results_dir_path = './results/'
        args.ifload = True
        args.fea_flag = True
        args.dataclean = 'SF'

        args.feature_size = 83
        args.num_classes = 6
        args.save_name = args.corruption_type + '_' + str(int(args.corruption_prob * 10)) + '_0' + \
                         str(args.meta_size_prob).split('.')[1]
        args.metadata_name = 'meta_data_' + args.meta_type + '_' + str(args.meta_size_prob).replace('.','') + '.npz'

        args.epochs = 110
        args.epochs_wp = 80
        args.epochs_model_wp = 70
        args.epochs_aaec_wp = 70
        args.epochs_g_delay = 70

        args.lr = 1e-2
        args.batch_size = 100
        args.weight_decay = 5e-4
        args.momentum = 0.9

        args.lr_aaec = 1e-2
        args.batch_size_aaec = 5000
        args.weight_decay_aaec = 5e-4
        args.momentum_aaec = 0.9

        args.dc_strategy_type = 'sf'
        args.dc_delay = 8
        args.dc_replace = True

        args.categories = ['Benign', 'BruteForce', 'DoS Hulk', 'Infilteration', 'HTTP DDoS', 'Bot']
        return args


def config_warmup(args):
    if args.dataset == 'ids18':
        # ===================================
        args.expriment_type = 'main'    # main, metasize, unannotated
        args.warmup_dir_path = './results/models/warmup/IDS18'
        args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_50r_33ep.pth'
        args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'

        # change in different experiments
        if args.expriment_type == 'main':
            args.corruption_type = 'sy'  # sy, asy,
            args.corruption_prob = 0.7

            args.meta_type = 'random'
            args.meta_size_prob = 0.01
            args.dc_expend_ratio = 0.2
            args.ceweight = [0.5, 3.6, 1.1, 3.5, 0.5, 2]

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_70r_21ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'

        elif args.expriment_type == 'metasize':
            args.corruption_type = 'asy'  # asy, sy
            args.corruption_prob = 0.5

            args.meta_type = 'balance'
            args.meta_size_prob = 500    #500, 100, 50
            args.dc_expend_ratio = 0.1
            args.ceweight = [1, 1, 1, 1, 1, 1]
            # args.ceweight = [0.5, 3.6, 1.1, 3.5, 0.5, 2]

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_50r_33ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_balance_500_85.pth'

        elif args.expriment_type == 'unannotated':
            args.corruption_type = 'ask'  # ask
            args.corruption_prob = 0.3  # 0.3  0.1

            args.meta_type = 'balance'
            args.meta_size_prob = 500
            args.dc_expend_ratio = 0.2

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_ask_3r_14ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_balance_500_85.pth'

            if args.corruption_prob == 0.1:
                args.ceweight = [1, 3.6, 1, 1, 1, 1]
            elif args.corruption_prob == 0.3:
                args.ceweight = [1, 1, 1, 3.5, 1, 1]
        # ====================================

        args.data_path = './data'
        args.results_dir_path = './results/'
        args.ifload = False
        args.fea_flag = True
        args.dataclean = 'SF'

        args.feature_size = 76
        args.num_classes = 6
        args.metadata_name = 'meta_data_' + args.meta_type + '_' + str(args.meta_size_prob).replace('.','') + '.npz'

        args.epochs = 79
        args.epochs_wp = 80
        args.epochs_model_wp = 80
        args.epochs_aaec_wp = 130
        args.epochs_g_delay = 70

        args.lr = 1e-1
        args.batch_size = 100
        args.weight_decay = 5e-4
        args.momentum = 0.9

        args.lr_aaec = 1e-1
        args.batch_size_aaec = 5000
        args.weight_decay_aaec = 5e-4
        args.momentum_aaec = 0.9

        args.dc_strategy_type = 'sf'
        args.dc_delay = 8
        args.dc_replace = True

        args.categories = ['Benign', 'BruteForce', 'DoS Hulk', 'Infilteration', 'HTTP DDoS', 'Bot']
        return args


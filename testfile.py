import os
import pickle
import argparse
import torch
import numpy as np
import dataset as module_data
import model_metric.loss as module_loss
import model_metric.metric as module_metric
from tqdm import tqdm
from model_metric.rec_metric import test_one_disease
from model import AFStroke as module_arch
from parse_config import ConfigParser

SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# "data_loader": {
#         "type": "PathDataLoader",
#         "args":{
#             "data_dir": "/home/zhiheng/cvd/zhiheng_network/",
#             "batch_size": 140,
#             "oversampling_factor": 0.7,
#             "undersampling_factor": 0.3,
#             "max_path_length": 8,
#             "max_path_num": 128,
#             "max_drug_disease_interaction": 10,
#             "max_disease_num": 30,
#             "max_stroke_path_num": 10,
#             "random_state": 30,
#             "recreate": false,
#             "use_disease_seed": true,
#             "shuffle": true,
#             "validation_split": 0.2,
#             "test_split": 0.1,
#             "num_workers": 2,
#             "oversampling": false,
#             "undersampling": false,
#             "partial_pair": false
#
#         }
#     },
def main(config):
    logger = config.get_logger('train')
    new_test = False
    save_flag = False
    # setup data_loader instances
    # data_loader = config.init_obj('data_loader', module_data)
    if new_test==False:
        data_loader = module_data.PathDataLoader(
            data_dir=config['data_loader']['args']['data_dir'],
            random_state_data_loader=config['data_loader']['args']['random_state_data_loader'],
            batch_size = config['data_loader']['args']['batch_size'],
            oversampling_factor = config['data_loader']['args']['oversampling_factor'],
            undersampling_factor= config['data_loader']['args']['undersampling_factor'],
            max_path_length=config['data_loader']['args']['max_path_length'],
            max_path_num=config['data_loader']['args']['max_path_num'],
            max_drug_disease_interaction=config['data_loader']['args']['max_drug_disease_interaction'],
            max_disease_num=config['data_loader']['args']['max_disease_num'],
            max_stroke_path_num=config['data_loader']['args']['max_stroke_path_num'],
            random_state=config['data_loader']['args']['random_state'],
            recreate=config['data_loader']['args']['recreate'],
            use_disease_seed=config['data_loader']['args']['use_disease_seed'],
            shuffle=config['data_loader']['args']['shuffle'],
            validation_split=config['data_loader']['args']['validation_split'],
            test_split=config['data_loader']['args']['test_split'],
            num_workers=config['data_loader']['args']['num_workers'],
            oversampling=False,
            undersampling=False,
            partial_pair=config['data_loader']['args']['partial_pair']

        )
        valid_data_loader = data_loader.split_dataset(valid=True)
        test_data_loader = data_loader.split_dataset(test=True)

    elif new_test==True:
        data_loader = module_data.PathDataLoader(
            data_dir=config['data_loader']['args']['data_dir'],
            random_state_data_loader=config['data_loader']['args']['random_state_data_loader'],
            batch_size=135,

            # batch_size=config['data_loader']['args']['batch_size'],
            oversampling_factor=config['data_loader']['args']['oversampling_factor'],
            undersampling_factor=config['data_loader']['args']['undersampling_factor'],
            max_path_length=config['data_loader']['args']['max_path_length'],
            max_path_num=config['data_loader']['args']['max_path_num'],
            max_drug_disease_interaction=config['data_loader']['args']['max_drug_disease_interaction'],
            max_disease_num=config['data_loader']['args']['max_disease_num'],
            max_stroke_path_num=config['data_loader']['args']['max_stroke_path_num'],
            random_state=config['data_loader']['args']['random_state'],
            recreate=config['data_loader']['args']['recreate'],
            use_disease_seed=config['data_loader']['args']['use_disease_seed'],
            shuffle=config['data_loader']['args']['shuffle'],
            validation_split=0,
            test_split=0,
            num_workers=config['data_loader']['args']['num_workers'],
            oversampling=False,
            undersampling=False,
            partial_pair=config['data_loader']['args']['partial_pair'],
            testing=True

        )



    adj = data_loader.get_sparse_adj()
    node_num = data_loader.get_node_num()
    type_num = data_loader.get_type_num()
    # new_test_data_loader = module_data.PathDataLoader(
    #     data_dir=config['data_loader']['args']['data_dir'],
    #     random_state_data_loader=config['data_loader']['args']['random_state_data_loader'],
    #     batch_size = config['data_loader']['args']['batch_size'],
    #     oversampling_factor = config['data_loader']['args']['oversampling_factor'],
    #     undersampling_factor= config['data_loader']['args']['undersampling_factor'],
    #     max_path_length=config['data_loader']['args']['max_path_length'],
    #     max_path_num=config['data_loader']['args']['max_path_num'],
    #     max_drug_disease_interaction=config['data_loader']['args']['max_drug_disease_interaction'],
    #     max_disease_num=config['data_loader']['args']['max_disease_num'],
    #     max_stroke_path_num=config['data_loader']['args']['max_stroke_path_num'],
    #     random_state=config['data_loader']['args']['random_state'],
    #     recreate=config['data_loader']['args']['recreate'],
    #     use_disease_seed=config['data_loader']['args']['use_disease_seed'],
    #     shuffle=config['data_loader']['args']['shuffle'],
    #     validation_split=0,
    #     test_split=0,
    #     num_workers=config['data_loader']['args']['num_workers'],
    #     oversampling=False,
    #     undersampling=False,
    #     partial_pair=config['data_loader']['args']['partial_pair'],
    #     testing=True
    #
    # )


    # build model architecture, then print to console
    # model = module_arch(node_num=node_num,
    #                     type_num=type_num,
    #                     adj=adj,
    #                     emb_dim=config['arch']['args']['emb_dim'],
    #                     gcn_layersize=config['arch']['args']['gcn_layersize'],
    #                     dropout=config['arch']['args']['dropout'],
    #                     mlpdropout=config['arch']['args']['mlpdropout'],
    #                     mlpbn=config['arch']['args']['mlpbn'],
    #                     alpha=config['arch']['args']['alpha'],
    #                     weight_loss_pos = weight_loss_pos)
    model = module_arch(node_num=node_num,
                        type_num=type_num,
                        number_of_heads=config['arch']['args']['number_of_heads'],
                        adj=adj,
                        emb_dim=config['arch']['args']['emb_dim'],
                        gcn_layersize=config['arch']['args']['gcn_layersize'],
                        dropout=config['arch']['args']['dropout'],
                        mlpdropout=config['arch']['args']['mlpdropout'],
                        mlpbn=config['arch']['args']['mlpbn'],
                        alpha=config['arch']['args']['alpha'],
                        gcn=config['arch']['args']['gcn'],
                        num_of_direction=config['arch']['args']['num_of_direction']
                        )
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    weight_loss_0 = config['weight_loss_0']
    weight_loss_1 = config['weight_loss_1']
    weight_loss_pos = config['weight_loss_pos']

    # load trained model
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    torch.cuda.set_device(1)
    # prepare model for testing
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device( 'cpu')

    model = model.to(device)
    model.eval()

    def inference(temp_data_loader, display_string, save_file_name, save_or_not = True):
        total_loss = 0.0
        total_metrics = torch.zeros(len(metrics))
        save_dict = {'input': [], 'output': [], 'true_label': [],
                     'drug_node_weight': [], 'drug_path_weight': [],
                     'disease_node_weight':[], 'disease_path_weight':[],
                     'output_embedding': [],'patid':[],'stroke_input':[],
                     "output_weight":[],"output_bias":[]

                     }
        with torch.no_grad():
            # for batch_idx, (drug, disease, path_feature, type_feature, lengths, mask, target) in enumerate(
            #         temp_data_loader):
            for batch_idx, (path_feature, type_feature, onehot_tensor, lengths, mask, target,
                        stroke_feature, stroke_type_feature, stroke_lengths_feature,
                        stroke_mask_feature,patid,sex_age) in enumerate(temp_data_loader):
            # for batch_idx, (path_feature, type_feature, lengths, mask, target,
            #                 stroke_feature, stroke_type_feature, stroke_lengths_feature,
            #                 stroke_mask_feature) in enumerate(temp_data_loader):
                # path_feature, type_feature = path_feature.to(device), type_feature.to(device)
                # mask, target = mask.to(device), target.to(device)
                # batch_idx = batch_idx.to_bytes(device)
                path_feature, type_feature = path_feature.to(device), type_feature.to(device)
                onehot_tensor,sex_age = onehot_tensor.to(device),sex_age.to(device)
                lengths, stroke_lengths_feature = lengths.to(device), stroke_lengths_feature.to(device)
                stroke_feature, stroke_type_feature = stroke_feature.to(device), stroke_type_feature.to(
                    device)
                mask, target, stroke_mask_feature,patid = mask.to(device), target.to(
                    device), stroke_mask_feature.to(device), patid.to(device)

                # output, node_weight_normalized, path_weight_normalized,disease_weight_normalized,output_embedding\
                #     = model(path_feature, type_feature, lengths, mask, stroke_feature
                #                                                                ,stroke_type_feature,stroke_lengths_feature
                #                                                                ,stroke_mask_feature,gcn=False)

                output, drug_node_weight_normalized, drug_path_weight_normalized, \
                stroke_node_weight_normalized, disease_weight_normalized, output_embedding,output_weight,\
                    output_bias= model(path_feature, type_feature, lengths, mask, stroke_feature
                                                                               ,stroke_type_feature,stroke_lengths_feature
                                                                               ,stroke_mask_feature,onehot_tensor,sex_age,gcn=False)

                loss = criterion(output, target,[weight_loss_0,weight_loss_1,weight_loss_pos])

                batch_size = path_feature.shape[0]
                total_loss += loss.item() * batch_size
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for i, metric in enumerate(metrics):
                    total_metrics[i] += metric(y_pred, y_true) * batch_size

                # for saving
                save_dict['input'].append(path_feature.cpu().detach().numpy())
                save_dict['output'].append(y_pred)
                save_dict['true_label'].append(y_true)
                save_dict['drug_node_weight'].append(drug_node_weight_normalized.cpu().detach().numpy())
                save_dict['drug_path_weight'].append(drug_path_weight_normalized.cpu().detach().numpy())
                save_dict['disease_node_weight'].append(stroke_node_weight_normalized.cpu().detach().numpy())
                save_dict['disease_path_weight'].append(disease_weight_normalized.cpu().detach().numpy())
                save_dict['patid'].append(patid.cpu().detach().numpy())
                save_dict['output_embedding'].append(output_embedding.cpu().detach().numpy())
                save_dict['stroke_input'].append(stroke_feature.cpu().detach().numpy())
                save_dict['output_weight'].append(output_weight.cpu().detach().numpy())
                save_dict['output_bias'].append(output_bias.cpu().detach().numpy())

            # save_dict['drug'].append(drug)
                # save_dict['disease'].append(disease)

        logger.info(display_string)
        n_samples = len(temp_data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples \
            for i, met in enumerate(metrics)})
        logger.info(log)
        if save_or_not == True:
            logger.info('Save predictions...')
            # savedir = os.path.join(config.save_dir,'retest_results', save_file_name)
            # print(savedir)
            print(f"file directory: {os.path.join(config.save_dir, 'retest_results',save_file_name)}")
            os.makedirs(os.path.join(config.save_dir, 'retest_results'), exist_ok=True)
            with open(os.path.join(config.save_dir,'retest_results', save_file_name), 'wb') as f:
                pickle.dump(save_dict, f)

    # Ks = config['Ks']

    # def drug_repurposing():
    #     result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
    #               'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    #     total_test_drug, total_test_disease = data_loader.get_recommendation_data()
    #
    #     y_pred_list, rating_dict_list, pos_test_drugs_list = [], [], []
    #     drug_used_list, disease_used = [], []
    #     for disease in tqdm(total_test_disease):
    #         try:
    #             path_feature, type_feature, lengths, mask, target, drug_used = data_loader.create_path_for_repurposing(
    #                 disease=disease, total_test_drug=total_test_drug)
    #             path_feature, type_feature, target = path_feature.to(device), type_feature.to(device), target.to(device)
    #             mask = mask.to(device)
    #             output, _, _ = model(path_feature, type_feature, lengths, mask, gcn=False)
    #             y_pred = torch.sigmoid(output)
    #         except:
    #             continue
    #
    #         y_pred = y_pred.cpu().detach().numpy()
    #         rating_dict = {drug_used[idx]: y_pred[idx] for idx in range(len(drug_used))}
    #         pos_test_drugs = data_loader.disease2drug_dict[disease]
    #
    #         disease_used.append(disease)
    #         drug_used_list.append(drug_used)
    #         y_pred_list.append(y_pred)
    #         rating_dict_list.append(rating_dict)
    #         pos_test_drugs_list.append(pos_test_drugs)
    #
    #         disease_metrics = test_one_disease(rating_dict=rating_dict,
    #                                            test_drugs=drug_used, pos_test_drugs=pos_test_drugs, Ks=Ks)
    #
    #         for metric_name, metric_array in result.items():
    #             result[metric_name] += disease_metrics[metric_name]
    #
    #     for metric_name, metric_array in result.items():
    #         result[metric_name] = result[metric_name] / len(disease_used)
    #
    #     logger.info('{} diesease have been tested.'.format(len(disease_used)))
    #     for idx, K in enumerate(Ks):
    #         log = {metric_name: metric_array[idx] for metric_name, metric_array in result.items() if
    #                metric_name != 'auc'}
    #         logger.info('Top {0} Predictions.'.format(K))
    #         logger.info(log)
    #
    #     logger.info('Save...')
    #     result_dict = {'y_pred': y_pred_list, 'disease': disease_used, 'drug': drug_used_list,
    #                    'rating_dict': rating_dict_list, 'pos_test_drugs': pos_test_drugs_list}
    #     with open(os.path.join(config.save_dir, 'result_dict.pkl'), 'wb') as f:
    #         pickle.dump(result_dict, f)

    if new_test==True:
        print("test_results_for_new_patients_with_depression_drugs_now")
        inference(temp_data_loader=data_loader, display_string='Train dataset', save_file_name='new_test_dict.pkl',save_or_not=save_flag)
    elif new_test==False:
        inference(temp_data_loader=data_loader, display_string='Train dataset', save_file_name='train_dict.pkl',save_or_not=save_flag)
        inference(temp_data_loader=valid_data_loader, display_string='Valid dataset', save_file_name='valid_save_dict.pkl',save_or_not=save_flag)
        inference(temp_data_loader=test_data_loader, display_string='Test dataset', save_file_name='test_save_dict.pkl',save_or_not=save_flag)
        # inference(test_data_loader=new_test_data_loader,display_string='new_test_dataset',save_file_name='new_test_dict.pkl')
    # drug_repurposing()
    """
    0709_030941
    """
if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')

    "0320_004050"
    "0319_161234"
    "0317_165126"
    "0315_163521"
    "0307_151721"
    "0306_140936"
    "0305_212114"
    "0305_200722"
    "0305_171353"
    "0305_154516"
    "0305_140542"
    "0305_025954"
    "0305_015908"
    "0304_225824"
    "0305_001106"
    "1220_130429"
    "1125_182104"
    "1117_094721"
    "1125_151528" "best" "test path 1125_160736"
    "1125_201806" "best" "test path ""1126_003849"
    file_name = "0322_203450"
    # test_one="/home/zhiheng/cvd/saved/models/AF_Stroke/1102_224005/config.json"
    config_paths = "/home/zhiheng/cvd/saved/models/AF_Stroke/"+file_name+"/config.json"
    model_paths = "/home/zhiheng/cvd/saved/models/AF_Stroke/"+file_name+"/model_best.pth"
    args.add_argument('-c', '--config', default=config_paths, type=str,
                      help='config file path (default: None)')
    # syr1 = "/home/zhiheng/cvd/saved/models/AF_Stroke/0825_175444/checkpoint-epoch8.pth"
    # syr2 = "/home/zhiheng/cvd/saved/models/AF_Stroke/1102_224005/model_best.pth"
    args.add_argument('-r', '--resume', default=model_paths, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)

    main(config)

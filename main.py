import logging
import sys
import argparse
import configparser
import torch
import threading

from legal_judgment_prediction.tools.analyze.analyze import analyze
from legal_judgment_prediction.tools.generate.generate import generate
from legal_judgment_prediction.tools.initialize import initialize_all
from legal_judgment_prediction.tools.train import train
from legal_judgment_prediction.tools.eval import eval
from legal_judgment_prediction.tools.serve.serve import serve_socket, serve_simple_IO
from line_bot.app import App_Thread

information = ' '.join(sys.argv)

def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='The path of config file', required=True)
    parser.add_argument('--mode', '-m', help='analyze, generate, train, eval or serve', required=True)
    parser.add_argument('--label', '-l', help='one_label or multi_labels (If mode is analyze or generate, this is required.)')
    parser.add_argument('--range', '-r', help='top_50_article or all (If mode is generate, this is required.)')
    parser.add_argument('--gpu', '-g', help='The list of gpu IDs (If mode is not analyze or generate, this is required.)')
    parser.add_argument('--use_checkpoint', '-uc', help='Use checkpoint (Ignore if do not use checkpoint)', action='store_true')
    parser.add_argument('--do_test', '-dt', help='Do test while training (Ignore if do not test while training)', action='store_true')
    parser.add_argument('--open_server', '-os', help='Open web server while serving (Ignore if do not open web server while serving', action='store_true')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    log_name = config.get('log', 'name')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    # precedent_analysis.log or bart.log or bert.log or mT5.log
    fh = logging.FileHandler(f'legal_judgment_prediction/logs/{log_name}', mode='a', encoding='UTF-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)
    
    logger.info(information)

    if args.mode == 'analyze':
        analyze(config, args.label)
    elif args.mode == 'generate':
        generate(config, args.label, args.range)
    else:
        gpu_list = []
        if args.gpu is not None:
            device_list = args.gpu.replace(' ', '').split(',')

            for device in range(0, len(device_list)):
                gpu_list.append(int(device))

        cuda_available = torch.cuda.is_available()

        logger.info(f'CUDA available: {str(cuda_available)}')

        if not cuda_available and len(gpu_list) > 0:
            logger.error('CUDA is not available but gpu_list is not empty.')
            raise Exception('CUDA is not available but gpu_list is not empty.')

        parameters = initialize_all(config, gpu_list, args.mode, args.use_checkpoint)

        if args.mode == 'train':
            train(parameters, config, gpu_list, args.do_test)
        elif args.mode == 'eval':  
            eval(parameters, config, gpu_list)
        elif args.mode == 'serve':
            if args.open_server == True:
                ljp_thread = threading.Thread(target=serve_socket, args=(parameters, config))
                ljp_thread.start()

                line_bot_thread = App_Thread(parameters)
                line_bot_thread.start()

                ljp_thread.join()

                line_bot_thread.shutdown()
                line_bot_thread.join()
            else:
                serve_simple_IO(parameters, config)
        else:
            logger.error('Invalid mode, please check again.')
            raise Exception('Invalid mode, please check again.')


if __name__ == '__main__':
    main()
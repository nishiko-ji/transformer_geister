import os

from log import Log


def make_data_txt(data_dir, data_name, log_dir, log_files):
    with open(f'{data_dir}{data_name}.txt', 'w', encoding='UTF-8') as f:
        for file in log_files:
            log = Log(log_dir+file)
            log.read_log()
            moves = ','.join(log.moves)
            f.write(f'{log.red_pos0} {log.red_pos1} {moves}\n')

def make_data_txt2(data_dir, data_name, log_dir, log_files):
    with open(f'{data_dir}{data_name}.txt', 'w', encoding='UTF-8') as f:
        for file in log_files:
            log = Log(log_dir+file)
            log.read_log()
            moves = ','.join(log.moves)
            label = []
            for l in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                if l in log.red_pos0:
                    label.append('r')
                else:
                    label.append('b')
            label2 = []
            for l in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                if l in log.red_pos1:
                    label2.append('r')
                else:
                    label2.append('b')
            print(label)
            print(label2)
            l = ','.join(label)
            l2 = ','.join(label2)
                
            # f.write(f'{log.red_pos0} {log.red_pos1} {moves}\n')
            print(f'{l},<sep>,{l2},<sep>,{moves}\n')
            f.write(f'{l},<sep>,{l2},<sep>,{moves}\n')

# def make_data_csv(data_dir, data_name, log_dir, log_files):
#     with open(f'{data_dir}{data_name}.csv', 'w', encoding='UTF-8') as f:
#         for file in log_files:
#             log = Log(log_dir+file)
#             log.read_log()
#             moves = ','.join(log.moves)
#             f.write(f'{log.red_pos0},{log.red_pos1},{moves}\n')

def main():
    # data_name = 'Naotti_Naotti'
    # data_name = 'Naotti_hayazashi'
    # data_name = 'hayazashi_Naotti'
    data_names = [
            'Naotti_hayazashi',
            'hayazashi_Naotti',
            'Naotti_Naotti',
            ]
    for data_name in data_names:
        log_dir = f'../log/{data_name}/log/'
        data_dir = '../data/'
        log_files = os.listdir(log_dir)
        make_data_txt(data_dir, data_name, log_dir, log_files)
        # make_data_csv(data_dir, data_name, log_dir, log_files)


if __name__ == '__main__':
    main()

import os
from csv import DictWriter


FIELD_NAMES = [
    'name', 'n_vertices', 'n_visibles', 'n_non-visibles', 'n_occluded', 'p_visibles', 'p_non-visibles', 'p_occluded',
]


def print_results(results):
    for key, value in results.items():
        print(key + ': ' + str(value))


def create_out_folder(result_path):
    if not os.path.isdir(result_path):
        print('Output folder %s not existed. Creating one.' % args.inf_result_folder)
        os.makedirs(args.inf_result_folder)


def create_out_file(result_path, result_file):

    create_out_folder(result_path)

    file_path = result_path + '/' + result_file
    if not os.path.isfile(file_path):
        with open(file_path, 'w+') as f:
            dw = DictWriter(
                f, delimiter=',', 
                fieldnames=FIELD_NAMES
            )
            dw.writeheader()

def _calculate_percentage(N, number):
    return round(float(number*100)/N, 2)

def save_results(name, n_vertices, n_visibles, n_invisibles, n_occluded, result_path, result_file='statistics.csv'):

    results = {
        'name': name, 
        'n_vertices': n_vertices, 
        'n_visibles': n_visibles, 
        'n_non-visibles': n_invisibles, 
        'n_occluded': n_occluded,  
        'p_visibles': _calculate_percentage(n_vertices, n_visibles), 
        'p_non-visibles': _calculate_percentage(n_vertices, n_invisibles), 
        'p_occluded': _calculate_percentage(n_vertices, n_occluded)
    }
    print_results(results)

    create_out_file(result_path, result_file)

    with open(result_path + '/' + result_file, 'a+') as f:
        dictwriter_object = DictWriter(f, fieldnames=FIELD_NAMES)
        dictwriter_object.writerow(results)
        f.close()

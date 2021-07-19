import argparse
import os


def main(input_file, output_path, human_labeling):

    import sys
    import csv
    import nltk
    from Parser import ParseTree, patterns
    from sklearn.metrics import cohen_kappa_score
    from tabulate import tabulate
    from statistics import mean

    labeling_exists = False
    if not os.path.isfile(input_file):
        raise RuntimeError("Input file path either doesn't exist or is not a file.")
    if not input_file.endswith('.txt'):
        raise RuntimeError('Unsupported format! Please provide input file as .txt!')
    if human_labeling is not None:
        labeling_exists = True
        if not os.path.isfile(human_labeling):
            raise RuntimeError(" Human labeling file path either doesn't exist or is not a file.")
        if not human_labeling.endswith('.csv'):
            raise RuntimeError('Unsupported format! Please provide human labeling file as .csv!')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(output_path):
        raise RuntimeError('Output path must be a directory')
    if not output_path.endswith('/'):
        output_path += '/'

    lal_parser_path = os.getcwd() + '/LAL-Parser/'
    os.system('python '
              + lal_parser_path + 'src_joint/main.py parse --contributions 0 --input-path '
              + input_file + ' --output-path-synconst '
              + output_path + 'output_synconst --output-path-syndep '
              + output_path + 'output_syndephead --output-path-synlabel '
              + output_path + 'output_syndeplabel --embedding-path '
              + lal_parser_path + 'data/glove.gz --model-path-base ' + lal_parser_path + 'best_parser.pt')

    with open(input_file) as file:
        requirements = []
        for line in file:
            requirements.append(line)
    actual_reqs = []
    for requirement in requirements:
        actual_reqs.append(nltk.word_tokenize(requirement))

    with open(output_path+'output_syndephead_0.txt') as file:
        reader = csv.reader(file)
        dep_heads = []
        for line in reader:
            head_list = []
            for item in line:
                head_list.append(int(''.join(e for e in item if e.isnumeric())))
            dep_heads.append(head_list)

    with open(output_path+'output_syndeplabel_0.txt') as file:
        reader = csv.reader(file)
        dep_labels = []
        for line in reader:
            label_list = []
            for item in line:
                label_list.append(''.join(e for e in item if e.isalnum()))
            dep_labels.append(label_list)

    parse_trees = []
    for dep_head, dep_label, text in zip(dep_heads, dep_labels, actual_reqs):
        try:
            parse_trees.append(ParseTree(dep_head, dep_label, text))
        except IndexError:
            print(str(dep_head) + "\n" + str(dep_label) + "\n" + str(text), file=sys.stderr)

    count = 0
    for tree in parse_trees:
        tree.clean_labelling()

    for pattern in patterns:
        for tree in parse_trees:
            if not tree.pattern_applied:
                success, output = tree.apply_pattern(**pattern)
                if success:
                    count += 1
    print("Number of patterns used:", str(len(patterns)))
    print("Labeled instances: " + str(count / len(parse_trees) * 100) + "%")
    print("No fitting labeling was found for", str(len(parse_trees) - count), "sentences")

    with open(output_path+'automated_labels.csv', 'w') as file:
        file.write('ID,labeling\n')
        for instance_no, tree in enumerate(parse_trees):
            if tree.pattern_applied:
                file.write(str(instance_no)+', ')
                for label in tree.get_current_labelling():
                    file.write(label+' ')
                file.write('\n')

    if labeling_exists:
        with open(human_labeling, 'r') as file:
            reader = csv.DictReader(file)
            labels = {}
            for line in reader:
                labels[line['ID']] = line['labeling'].split()

        adjusted_labels = {}
        for instance_no, labelling in labels.items():
            new_labelling = []
            for label in labelling:
                if label == '1':
                    new_labelling.append('ent1')
                elif label == '2':
                    new_labelling.append('ent2')
                elif label == 'c':
                    new_labelling.append('cond')
                elif label == 'O':
                    new_labelling.append('O')
                elif label == 'r':
                    new_labelling.append('rel')
                else:
                    print('Error! Unknown label! ', label, file=sys.stderr)
                    break
            adjusted_labels[int(instance_no)] = new_labelling
        labels = adjusted_labels

        hl_list = []
        al_list = []
        human_labels = []
        automated_labels = []
        for instance_no, labelling in labels.items():
            hl_list.append((instance_no, labelling))
            human_labels.extend(labelling)
            automated_labels.extend(parse_trees[instance_no].get_current_labelling())
            al_list.append(parse_trees[instance_no].get_current_labelling())
            if len(human_labels) - len(automated_labels) != 0:
                raise RuntimeError("Human labeling of ID ", instance_no, " doesn't match length of original sentence!")

        kappa_scores = []
        ent1_kappa = []
        ent2_kappa = []
        rel_kappa = []
        cond_kappa = []
        for instance_no, labelling in labels.items():
            auto_labelling = parse_trees[instance_no].get_current_labelling()
            kappa_scores.append(cohen_kappa_score(labelling, auto_labelling))
            ent1_kappa.append(cohen_kappa_score(['O' if hl != 'ent1' else 'ent1' for hl in labelling],
                                                ['O' if al != 'ent1' else 'ent1' for al in auto_labelling]))
            if 'ent2' in labelling or 'ent2' in auto_labelling:
                ent2_kappa.append(cohen_kappa_score(['O' if hl != 'ent2' else 'ent2' for hl in labelling],
                                                    ['O' if al != 'ent2' else 'ent2' for al in auto_labelling]))
            rel_kappa.append(cohen_kappa_score(['O' if hl != 'rel' else 'rel' for hl in labelling],
                                               ['O' if al != 'rel' else 'rel' for al in auto_labelling]))
            if 'cond' in labelling or 'cond' in auto_labelling:
                cond_kappa.append(cohen_kappa_score(['O' if hl != 'cond' else 'cond' for hl in labelling],
                                                    ['O' if al != 'cond' else 'cond' for al in auto_labelling]))

        only_ent1_human = ['O' if hl != 'ent1' else 'ent1' for hl in human_labels]
        only_ent1_auto = ['O' if al != 'ent1' else 'ent1' for al in automated_labels]
        only_ent2_human = ['O' if hl != 'ent2' else 'ent2' for hl in human_labels]
        only_ent2_auto = ['O' if al != 'ent2' else 'ent2' for al in automated_labels]
        only_cond_human = ['O' if hl != 'cond' else 'cond' for hl in human_labels]
        only_cond_auto = ['O' if al != 'cond' else 'cond' for al in automated_labels]
        only_rel_human = ['O' if hl != 'rel' else 'rel' for hl in human_labels]
        only_rel_auto = ['O' if al != 'rel' else 'rel' for al in automated_labels]

        table = tabulate([['All labels', mean(kappa_scores), cohen_kappa_score(human_labels, automated_labels)],
                          ['rel only', mean(rel_kappa), cohen_kappa_score(only_rel_auto, only_rel_human)],
                          ['ent1 only', mean(ent1_kappa), cohen_kappa_score(only_ent1_auto, only_ent1_human)],
                          ['ent2 only', mean(ent2_kappa), cohen_kappa_score(only_ent2_auto, only_ent2_human)],
                          ['cond only', mean(cond_kappa), cohen_kappa_score(only_cond_auto, only_cond_human)]],
                         ['Labels considered', 'Sentence Average', 'Overall'], floatfmt='.3f', tablefmt='psql')
        print(table)


parser = argparse.ArgumentParser()
parser.add_argument('--input-file', '-i', required=True, help="Path to the input file. Must be provided in .txt. "
                                                              "Each line should be exactly one sentence.")
parser.add_argument('--output-dir', '-o', required=True, help="Path of output directory")
parser.add_argument('--human_labeling', '-l', help="Path to the human labeling file for "
                                                   "Cohen's kappa calculation. Must be provided as .csv "
                                                   "with 'ID' and 'labeling' columns. "
                                                   "Each line should correspond to one sentence. '1' stands for ent1,"
                                                   "'2' for ent2, 'c' for cond and 'r' for rel. Each label is separated"
                                                   " by spaces.")

if __name__ == "__main__":
    abs_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(abs_path)
    os.chdir(dir_name)
    arguments = parser.parse_args()
    main(arguments.input_file, arguments.output_dir, arguments.human_labeling)

from __future__ import annotations
from typing import Optional, List, Tuple
from nltk.tokenize.treebank import TreebankWordDetokenizer


class ParseTree:

    def __init__(self, dependency_heads: List[int], dependency_labels: List[str], sentence: List[str]):
        self.sentence = TreebankWordDetokenizer().detokenize(sentence)
        self.pattern_applied = False
        self.nodes = []
        # there should be only one root which has
        root_index = dependency_heads.index(0)
        for index, word in enumerate(sentence):
            self.nodes.append(Node(word, index, dependency_labels[index]))

        self.root = self.nodes[root_index]
        for index, node in enumerate(self.nodes):
            if dependency_heads[index] != 0:
                node.set_parent(self.nodes[dependency_heads[index] - 1])
                self.nodes[dependency_heads[index] - 1].add_children(node)

    def __str__(self):
        output = self.sentence + '\n'
        output += str(self.root) + '\n'
        output += self.root.print_children(0)
        return output

    def apply_pattern(self, pattern: List[Tuple[str, List[str], bool]], final: bool) -> Tuple[bool, List[str]]:
        # the pattern should look the following: each dictionary key defines the entity to be applied
        # the value is a list of dependencies we traverse down the parse tree
        # the last dependent and, if only_root (the boolean in the dict) is False,
        # all its children will be tagged with the key
        previous_labels = self.get_current_labelling()
        for entity, path, only_root in pattern:
            if path[0] != 'root':
                if path[0].split('=')[0] != 'root':
                    raise ValueError('Pattern must start from the root! Your pattern starts with ' + path[0])
            current_nodes = [self.root]
            if len(path) > 1:
                for step in path[1:]:
                    current_nodes_temp = []
                    # keep parent if it does NOT contain the given dependency
                    if step.startswith('!'):
                        for node in current_nodes:
                            include = True
                            for rel in node.children:
                                if rel == step[1:]:
                                    include = False
                                    break
                            if include:
                                current_nodes_temp.append(node)

                    else:
                        for node in current_nodes:
                            if step == '..':
                                current_nodes_temp.append(node.parent)
                            elif '=' not in step:
                                if step in node.children:
                                    current_nodes_temp.extend(node.children[step])
                            else:
                                step_split = step.split('=')
                                if step_split[0] in node.children:
                                    for child in node.children[step_split[0]]:
                                        if child.word.lower() == step_split[1].lower():
                                            current_nodes_temp.append(child)
                    current_nodes = current_nodes_temp
                    if not current_nodes:
                        for node, previous_label in zip(self.nodes, previous_labels):
                            node.set_pattern_label(previous_label, True)
                        return False, []
            for node in current_nodes:
                node.set_pattern_label(entity, only_root)

            # if there are no more patterns allowed to be applied, set to True
        if final:
            self.pattern_applied = final
        return final, [node.pattern_label for node in self.nodes]

    def get_current_labelling(self) -> List[str]:
        return [node.pattern_label for node in self.nodes]

    def clean_labelling(self):
        for node in self.nodes:
            node.pattern_label = 'O'
        self.pattern_applied = False


class Node:

    def __init__(self, word: str, index: int, label: Optional[str] = None):
        if label != 'root':
            self.is_root = False
        else:
            self.is_root = True

        self.parent = None
        self.label = label
        self.index = index
        self.word = word
        self.children = {}
        # set all pattern labels to "O" for "Outside" in IO tagging
        self.pattern_label = "O"

    def __str__(self):
        return self.label + ": '" + self.word + "' [" + str(self.index) + "]"

    def add_children(self, node: Node):
        if node.label in self.children.keys():
            self.children[node.label].append(node)
        else:
            self.children[node.label] = [node]

    def set_parent(self, node: Node):
        self.parent = node

    def set_pattern_label(self, pattern_label: str, only_root: bool):
        self.pattern_label = pattern_label
        if not only_root:
            for children in self.children.values():
                for child in children:
                    child.set_pattern_label(pattern_label, only_root)

    def root(self) -> bool:
        return self.is_root

    def print_children(self, indent_level: int) -> str:
        output = ""
        indent = ""
        for i in range(indent_level):
            indent += '      '

        for children in self.children.values():
            for child in children:
                output += indent + "|---> " + str(child) + '\n'
                output += child.print_children(indent_level + 1)
        return output


patterns = [
    {
        'pattern': [
            ('rel', ['root', 'prt'], True)
        ],
        'final': False},
    {
        'pattern': [
            ('rel', ['root', 'neg'], True)
        ],
        'final': False},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('cond', ['root', '!dobj', '!xcomp', '!prep', 'advcl'], False),
            ('cond', ['root', 'prep', 'advmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False),
            ('cond', ['root', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False),
            ('cond', ['root', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'prep'], False),
            ('cond', ['root', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False),
            ('cond', ['root', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False),
            ('cond', ['root', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'prep'], False),
            ('cond', ['root', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'prep', 'pobj'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'prep', 'pcomp'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'advmod'], False),
            ('cond', ['root', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'advmod'], False),
            ('cond', ['root', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'prep'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'advmod'], False),
            ('cond', ['root', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'advmod'], False),
            ('cond', ['root', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'advmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'advmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'prep'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'advmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'advmod'], False)
        ],
        'final': True},
    # {
    # 'pattern': [
    #    ('rel', ['root', 'dobj'], False),
    #    ('ent2', ['root','dobj', 'prep', 'pobj'], False),
    #    ('ent1', ['root', 'nsubj'], False)
    # ],
    # 'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'ccomp'], True),
            ('ent2', ['root', 'ccomp', 'nsubj'], False),
            ('cond', ['root', 'ccomp', 'nsubj', 'partmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'ccomp', 'dobj'], False),
            ('cond', ['root', 'ccomp', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'ccomp'], True),
            ('ent2', ['root', 'ccomp', 'nsubj'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'ccomp', 'dobj'], False),
            ('cond', ['root', 'ccomp', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'ccomp'], False),
            ('ent2', ['root', 'ccomp', 'nsubjpass'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'prep', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'ccomp'], True),
            ('ent2', ['root', 'ccomp', 'nsubj'], False),
            ('cond', ['root', 'ccomp', 'nsubj', 'partmod'], False),
            ('ent1', ['root', 'nsubj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'ccomp'], True),
            ('ent2', ['root', 'ccomp', 'nsubj'], False),
            ('ent1', ['root', 'nsubj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False),
            ('cond', ['root', 'partmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False),
            ('cond', ['root', 'partmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'prep'], False),
            ('cond', ['root', 'partmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'partmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'dobj'], False),
            ('cond', ['root', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', 'prep', 'pcomp'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep', 'pcomp', 'dobj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!nsubj', '!dobj', 'xcomp'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('ent2', ['root', 'xcomp', 'dobj'], False),
            ('cond', ['root', 'xcomp', 'dobj', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!nsubj', '!dobj', 'xcomp'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('ent2', ['root', 'xcomp', 'dobj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!dobj', 'xcomp'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'xcomp', 'dobj'], False),
            ('cond', ['root', 'xcomp', 'dobj', 'prep'], False),
            ('cond', ['root', 'xcomp', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!dobj', 'xcomp'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'xcomp', 'dobj'], False),
            ('cond', ['root', 'xcomp', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('ent1', ['root', 'nsubjpass'], False),
            ('rel', ['root', 'prep', 'pobj', 'dep'], False),
            ('ent2', ['root', 'prep', 'pobj', 'dep', 'nsubj'], False),
            ('cond', ['root', 'prep', 'pobj', 'dep', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('ent2', ['root', 'nsubj'], False),
            ('rel', ['root', 'xcomp'], True),
            ('ent1', ['root', 'xcomp', 'prep', 'pobj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'cop'], True),
            ('ent1', ['root', '!xcomp', '!dobj', '!prep', 'nsubj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('ent1', ['root', 'nsubj'], False),
            ('rel', ['root', 'xcomp'], True),
            ('ent2', ['root', '!dobj', 'xcomp', 'dobj'], False),
            ('cond', ['root', 'xcomp', 'dobj', 'prep'], False),
            ('cond', ['root', 'xcomp', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('ent1', ['root', 'nsubj'], False),
            ('rel', ['root', 'xcomp'], True),
            ('ent2', ['root', '!dobj', 'xcomp', 'dobj'], False),
            ('cond', ['root', 'xcomp', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('ent1', ['root', 'nsubj'], False),
            ('rel', ['root', 'xcomp'], True),
            ('ent2', ['root', '!dobj', 'xcomp', 'dobj'], False),
            ('cond', ['root', 'xcomp', 'dobj', 'prep'], False),
            ('cond', ['root', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('ent1', ['root', 'nsubj'], False),
            ('rel', ['root', 'xcomp'], True),
            ('ent2', ['root', '!dobj', 'xcomp', 'dobj'], False),
            ('cond', ['root', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('ent1', ['root', 'nsubj'], False),
            ('rel', ['root', 'xcomp'], True),
            ('ent2', ['root', '!dobj', 'xcomp', 'dobj'], False),
            ('cond', ['root', 'xcomp', 'dobj', 'prep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('ent1', ['root', 'nsubj'], False),
            ('rel', ['root', 'xcomp'], True),
            ('ent2', ['root', '!dobj', 'xcomp', 'dobj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('cond', ['root', 'advcl'], False),
            ('cond', ['root', 'prep', 'dep'], False)
        ],
        'final': True},

    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('cond', ['root', '!dobj', '!xcomp', '!prep', 'advcl'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!nsubj', '!xcomp', '!prep', 'dobj', 'infmod'], True),
            ('ent1', ['root', 'dobj', 'infmod', 'dobj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!dobj', '!xcomp', '!prep'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'advcl'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!dobj', '!xcomp', '!prep'], True),
            ('rel', ['root', 'dep'], False),
            ('ent1', ['root', 'nsubj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'dobj'], False),
            ('cond', ['root', 'csubj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!dobj'], True),
            ('ent1', ['root', 'csubj'], True),
            ('ent1', ['root', 'csubj', 'dobj'], False),
            ('cond', ['root', 'advmod'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!dobj'], True),
            ('ent1', ['root', 'csubj'], True),
            ('cond', ['root', 'advmod'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'amod'], False),
            ('ent1', ['root', 'prep=of', 'pobj'], False),
            ('cond', ['root', 'rcmod'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'xcomp'], True),
            ('rel', ['root', 'xcomp', 'xcomp'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'xcomp', 'nsubj'], False),
            ('cond', ['root', 'xcomp', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('cond', ['root', 'nsubjpass', 'rcmod'], False),
            ('ent2', ['root', 'prep=to', 'pobj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep=to', 'pobj'], False),
            ('cond', ['root', 'advcl'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('ent2', ['root', 'prep=to', 'pobj'], False),
            ('cond', ['root', 'prep=at'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('ent2', ['root', 'prep=as', 'pobj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('ent2', ['root', 'prep=in', 'pobj'], False),
            ('cond', ['root', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('ent2', ['root', 'prep=in', 'pobj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('ent2', ['root', 'prep=to', 'pobj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep=on', 'pobj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'cop'], True),
            ('rel', ['root', 'xcomp'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'xcomp', '!dobj', '!nsubj', 'advmod'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'nsubjpass'], False),
            ('ent1', ['root', 'prep=by', 'pobj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('cond', ['root', 'prep=upon'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root=capable', 'prep=of', 'pcomp'], True),
            ('rel', ['root', 'prep=of', 'pcomp', 'cop'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep=of', 'pcomp', 'prep=with', 'pobj'], False),
            ('cond', ['root', 'prep=of', 'prep=across'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!dobj'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('cond', ['root', 'prep=in', 'pobj=case', '..'], False),
            ('cond', ['root', 'advmod'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!dobj', '!xcomp', '!prep'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('cond', ['root', 'nsubjpass', 'amod'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('ent2', ['root', 'prep=onto', 'pobj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root=capable', 'prep=of', 'pcomp'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep=of', 'pcomp', 'prep=in', 'pobj'], False),
            ('cond', ['root', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'xcomp'], True),
            ('ent1', ['root', '!dobj', '!prep', 'nsubj'], False),
            ('ent2', ['root', 'xcomp', 'nsubj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'xcomp'], True),
            ('ent1', ['root', '!dobj', 'nsubj'], False),
            ('ent2', ['root', 'xcomp', 'nsubj'], False),
            ('cond', ['root', 'prep=during'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', '!xcomp', '!dobj', '!prep', 'nsubjpass'], False),
            ('cond', ['root', 'ccomp'], False)
        ],
        'final': True},

    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'neg'], True),
            ('cond', ['root', 'advcl'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep=into', 'pobj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep=for', 'pobj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('cond', ['root', 'prep=to', 'advmod=prior', '..'], False),
            ('ent1', ['root', '!dobj', 'nsubj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep=with', 'pobj'], False),
            ('cond', ['root', 'advcl'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('ent2', ['root', 'prep=for', 'pobj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep=as', 'pobj'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'prep'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'dobj=capability'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('rel', ['root', 'dobj=capability'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dep', 'mark=if'], False),
            ('cond', ['root', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dep', 'mark=if'], False),
            ('cond', ['root', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'prep'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dep', 'mark=if'], False),
            ('cond', ['root', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dep', 'mark=if'], False),
            ('cond', ['root', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'prep'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('ent1', ['root', 'nsubj'], False),
            ('cond', ['root', 'dobj', 'dep'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!dobj', '!xcomp'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('cond', ['root', 'prep=during'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubjpass'], False),
            ('cond', ['root', 'prep=before'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', 'nsubj'], False),
            ('ent2', ['root', 'prep', 'pobj'], False),
            ('cond', ['root', 'prep', '!pobj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root', '!nsubj', '!nsubjpass', '!xcomp', '!prep'], True),
            ('ent1', ['root', 'dobj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent1', ['root', '!xcomp', '!dobj', '!prep', 'nsubjpass'], False),
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'rcmod'], False),
            ('ent1', ['root', 'nsubj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'infmod'], False),
            ('ent1', ['root', 'nsubj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('cond', ['root', 'dobj', 'prep'], False),
            ('ent1', ['root', 'nsubj'], False)
        ],
        'final': True},
    {
        'pattern': [
            ('rel', ['root'], True),
            ('ent2', ['root', 'dobj'], False),
            ('ent1', ['root', 'nsubj'], False)
        ],
        'final': True},
]

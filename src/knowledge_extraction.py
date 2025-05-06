import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.tree import DecisionTreeClassifier, _tree

class REMDExtractor:
    def __init__(self, model, feature_names, max_depth=4, input_dim=None):
        """
        model:         Trained tf.keras.Sequential or Functional model.
        feature_names: List[str] of original input feature names.
        max_depth:     Max depth for induced decision trees.
        input_dim:     Required if model.input is undefined (number of features).
        """
        self.model = model
        self.feature_names = feature_names
        self.max_depth = max_depth

        # Determine input dimension
        if input_dim is None:
            # Try to infer from model config (Sequential)
            try:
                first_layer = self.model.layers[0]
                input_dim = first_layer.input_shape[1]
            except:
                raise ValueError("input_dim not provided and could not infer from model.")
        self.input_dim = input_dim

        # Build a new Functional subgraph to extract hidden layer outputs
        inp = Input(shape=(self.input_dim,), name="extraction_input")
        x = inp
        hidden_outputs = []
        for layer in self.model.layers:
            x = layer(x)
            # capture each hidden Dense (exclude final output)
            if isinstance(layer, Dense) and layer != self.model.layers[-1]:
                hidden_outputs.append(x)

        # Create the activation model
        self.activation_model = Model(inputs=inp, outputs=hidden_outputs)

    def get_activations(self, X):
        """
        Returns a list of numpy arrays: activations for each hidden layer.
        """
        return self.activation_model.predict(X, verbose=0)

    def extract_tree_rules(self, tree, feature_labels):
        """
        Traverse a DecisionTreeClassifier to extract IF–THEN rules.
        """
        tree_ = tree.tree_
        rules = []

        def recurse(node, conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_labels[tree_.feature[node]]
                thr  = tree_.threshold[node]
                recurse(tree_.children_left[node],
                        conditions + [(name, '<=', thr)])
                recurse(tree_.children_right[node],
                        conditions + [(name, '>',  thr)])
            else:
                val = np.argmax(tree_.value[node])
                rules.append((conditions, val))

        recurse(0, [])
        text_rules = []
        for conds, val in rules:
            antecedent = ' & '.join(f"{n} {op} {thr:.4f}"
                                   for n, op, thr in conds)
            text_rules.append({'if': antecedent, 'then': int(val)})
        return text_rules

    def train_layer_trees(self, activations, preds):
        """
        Train one DecisionTree per class to predict membership from activations.
        """
        trees = {}
        for cls in np.unique(preds):
            dt = DecisionTreeClassifier(max_depth=self.max_depth)
            y_bin = (preds == cls).astype(int)
            dt.fit(activations, y_bin)
            trees[cls] = dt
        return trees

    def substitute_rules(self, trees, preds, layer_idx):
        """
        Extract and substitute rules at one layer.
        Returns updated preds and the new rule list.
        """
        n_feats = trees[next(iter(trees))].tree_.n_features
        feature_labels = [f"h{layer_idx}_n{i}" for i in range(n_feats)]

        new_rules = []
        for cls, tree in trees.items():
            for r in self.extract_tree_rules(tree, feature_labels):
                if r['then'] == 1:
                    new_rules.append({'if': r['if'], 'then': cls})

        new_preds = np.array([r['then'] for r in new_rules]) if new_rules else preds
        return new_preds, new_rules

    def translate_to_input(self, X, preds):
        """
        Train final trees on raw inputs to produce rules in original feature space.
        """
        final_trees = self.train_layer_trees(X, preds)
        input_rules = []
        for cls, tree in final_trees.items():
            for r in self.extract_tree_rules(tree, self.feature_names):
                if r['then'] == 1:
                    input_rules.append(r)
        return input_rules

    def extract(self, X):
        """
        1) Predict with MLP
        2) Extract hidden-layer activations
        3) Backward pass: train & substitute layer trees
        4) Translate final rules to input features
        Returns: List[Dict] of {'if': ..., 'then': ...} rules.
        """
        # 1) Initial predictions
        preds = np.argmax(self.model.predict(X, verbose=0), axis=1)

        # 2) Hidden-layer activations
        activations = self.get_activations(X)

        # 3) Backward pass through hidden layers
        for i in reversed(range(len(activations))):
            trees = self.train_layer_trees(activations[i], preds)
            preds, _ = self.substitute_rules(trees, preds, layer_idx=i)

        # 4) Final translation to original features
        return self.translate_to_input(X, preds)

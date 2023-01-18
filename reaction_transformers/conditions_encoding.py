import math
import json
import numpy as np

from tokenization import smi_tokenizer

class ConditionIDEncoder():
    def __init__(self, model_name, thresholds, id2role):
        
        self.model_name = model_name
        self.thresholds = thresholds
        self.id2role = id2role
        self.enriched_emb_vocab_size =  self._get_enrichment_size(model_name)
        self.config_path = f'../reaction_transformers/pretrained_models/transformers/bert_{model_name}/config.json'
        
        assert len(thresholds) < self.enriched_emb_vocab_size-5 , \
        f"The enrichment embedding size ({self.enriched_emb_vocab_size}) is not large enough for the total number of ID classes ({len(thresholds)+5}). \
        Reduce the number of thresholds or train a precursor model set to have a larger enrichment vocab size."
        
    def convert_conditions2class(self, condition_data):
        """
        Convert numerical/categorical condition data into enrichment class ids using user defined thresholds
        """
        enrich_classes = np.digitize(condition_data, self.thresholds)
        return list(map(lambda x:x+2, enrich_classes))
    
    def generate_equivalents_vector(self, row):
        """
        Generate a vector of categorical enrichment class ids with as many elements as the numbers of tokens in the reaction
        """
        reaction = row.reaction_smiles_sep
        enrichment_class = row.enrichment_class
        enrich_class_count = 0
        equivalents_vector = []
        for token in smi_tokenizer(reaction).split(" "):
            if token != " ":
                if token == ">>":
                    equivalents_vector.append(self.enriched_emb_vocab_size-3)
                    enrich_class_count = "product"
                else:
                    if enrich_class_count == "product":
                        equivalents_vector.append(self.enriched_emb_vocab_size-2)
                    else:
                        if token == "." or token == "^":
                            if token == ".":
                                equivalents_vector.append(self.enriched_emb_vocab_size-1)
                            if token == "^":
                                equivalents_vector.append(enrichment_class[enrich_class_count])
                            enrich_class_count += 1
                        else:
                            equivalents_vector.append(enrichment_class[enrich_class_count])
        return equivalents_vector
    
    def convert_reactant_id_2_moles(self, row):
        """
        Convert reactant ids (not the enriched embedding ids) to their respective moles quantity for a particular reaction
        """
        return [row[self.id2role[str(math.floor(r_id))]+"_moles"] for r_id in row.reactant_ids_order]
    
    def _get_enrichment_size(self, model_name):
        config_path = f'../reaction_transformers/pretrained_models/transformers/bert_{model_name}/config.json'
        with open(config_path) as f:
            config = json.load(f)
            return config['equivalents_vocab_size']
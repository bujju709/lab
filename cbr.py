import numpy as np 
 
class CaseBasedReasoning: 
    def __init__(self): 
        self.cases = [] 
 
    def add_case(self, features, label): 
        """Adds a new case with features and corresponding label.""" 
        self.cases.append({'features': np.array(features), 'label': label}) 
 
    def retrieve(self, query_features, k=1): 
        """Retrieves the k most similar cases based on Euclidean distance.""" 
        distances = [] 
        for case in self.cases: 
            distance = np.linalg.norm(case['features'] - np.array(query_features)) 
            distances.append((distance, case)) 
        distances.sort(key=lambda x: x[0]) 
        return [case for _, case in distances[:k]] 
 
    def predict(self, query_features, k=1): 
        """Predicts the label by majority voting among k nearest cases.""" 
        nearest_cases = self.retrieve(query_features, k) 
        labels = [case['label'] for case in nearest_cases] 
        return max(set(labels), key=labels.count) 
 
# Sample usage 
cbr = CaseBasedReasoning() 
 
# Adding some past cases 
cbr.add_case([1, 2], 'A') 
cbr.add_case([2, 3], 'A') 
cbr.add_case([3, 3], 'B') 
cbr.add_case([5, 4], 'B') 
cbr.add_case([3, 5], 'B') 
# Query case 
query = [2.5, 3] 
# Retrieve the most similar case 
similar_cases = cbr.retrieve(query, k=2) 
print("Most Similar Cases:", similar_cases) 
# Predict label based on nearest neighbors 
prediction = cbr.predict(query, k=2) 
print("Predicted Label:", prediction)

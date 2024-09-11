import pandas as pd
from datetime import datetime, timedelta
from genAI_course.evaluate import ModelEvaluator

# Create a DataFrame to store training results with SCD Type 2
training_results = pd.DataFrame(columns=[
    'model_id', 'algorithm', 'model_type', 'accuracy', 'f1_score', 'precision', 'recall', 'auc', 'start_date', 'end_date', 'is_current'
])

def insert_new_training_result(model_id, algorithm, model_type, accuracy, y_true, y_pred, y_prob=None):
    global training_results  # Declare the DataFrame as global

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')  # Yesterday as start_date

    # Evaluate model using ModelEvaluator
    evaluator = ModelEvaluator(y_true, y_pred, y_prob)
    f1 = evaluator.evaluate_f1_score()
    precision = evaluator.evaluate_precision()
    recall = evaluator.evaluate_recall()
    accuracy = evaluator.evaluate_accuracy()
    auc = evaluator.evaluate_auc()

    # Check if a current record exists for the given model_id
    existing_record = training_results[(training_results['model_id'] == model_id) & (training_results['is_current'] == True)]

    if not existing_record.empty:
        # If a current record exists, mark the existing record as inactive by setting the end_date and is_current = False
        training_results.loc[existing_record.index, 'end_date'] = current_time
        training_results.loc[existing_record.index, 'is_current'] = False

    # Convert the new result dictionary into a DataFrame
    new_result = pd.DataFrame([{
        'model_id': model_id,
        'algorithm': algorithm,
        'model_type': model_type,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'start_date': start_time,
        'end_date': None,  # Null since this is the current version
        'is_current': True
    }])

    # Use pd.concat to add the new result to the DataFrame
    training_results = pd.concat([training_results, new_result], ignore_index=True)

# Example usage with dummy true and predicted labels
# y_true = [0, 1, 1]  # True labels
y_true = y_train
y_pred = [0, 1, 1]  # Predicted labels
y_prob = [0.2, 0.9, 0.6]  # Predicted probabilities

insert_new_training_result(model_id=1, algorithm='Neural Network', model_type='classification', accuracy=0.85, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
insert_new_training_result(model_id=1, algorithm='Neural Network', model_type='classification', accuracy=0.88, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
insert_new_training_result(model_id=2, algorithm='Decision Tree', model_type='classification', accuracy=0.80, y_true=y_true, y_pred=y_pred)
insert_new_training_result(model_id=1, algorithm='Neural Network', model_type='classification', accuracy=accuracy_score(y_train, y_pred), y_true=y_true, y_pred=y_pred, y_prob=y_prob)
insert_new_training_result(model_id=2, algorithm='Decision Tree', model_type='classification', accuracy=accuracy_score(y_train, y_pred), y_true=y_true, y_pred=y_pred)

# View the updated DataFrame with SCD Type 2 applied
print(training_results)

# deep_learning_challenge
This repository contains code and analysis for a machine learning model aimed at predicting the success of Alphabet Soup-funded organizations using a neural network. The project involves data preprocessing, model building, optimization, and reporting.

# Analysis
### Data Preprocessing
1. The target variable (y) is application_df_with_dummies["IS_SUCCESSFUL"]. We are interested to determine whether a campaign was successful or not as a part of the model.
2. The features (x) are application_df_with_dummies.drop(['IS_SUCCESSFUL'], axis='columns').values. These include APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONs, ASK_AMT.
3. In the original model, EIN and NAME are idendification columns that were removed to simplify the model's feature set. NAME was added back into the optimized model as a feature as it is categorical in nature since some organizations have multiple campaigns defined in the dataset. SPECIAL_CONSIDERATIONS and STATUS were removed in the optimized model as their value to the feature set seemed to be low. After further training and testing, results confirmed that accuracy increase without these features set.

### Compiling, Training, and Evaluating the Model
1. The original model had 15 neurons, 2 hidden layers with RelU activation applied. The output layer uses a single neuron with sigmoid activation. The optimized model has over 80 neurons, 3 hidden layers with first/second being RelU and third being sigmoid activation. The output layer uses a single neuron with sigmoid activation.
2. The optimized model was able to achieve the targeted accuracy of over 75%. The optimized model achieved 79% accuracy.
3. The steps that were performed during model optimization included:
  - re-analysis of column value counts
  - determining additional columns to remove, SPECIAL_CONSIDERATIONS and STATUS
  - adjusting column values to categorical data, ASK_AMT
  - adding columns bad into the model as a feature, NAME
  - added one more hidden layer for 3 total
  - tested activation functions: attempted 3 RelU, 3 Sigmoid, 2 Sigmoid / 1 RelU, 2 RelU / 1 Sigmoid
  - increased neurons by a significant amount at each layer

### Summary
Overall, the model indicates that a successful campaign will be 79% likely if an organization runs at least 10 campaigns total under the application types of T3, T4, T5, T6, T19, 
has a classificiation of C1000, C1200, C2000, C2100 or C3000. Additional models that could be explored are smaller models segemented by ask amount of the campaign since smaller asks and larger asks may have different success rates and require analysis of the application types and classifications used in those categories. With this in mind, focused logistic regression models could be utilized to select specific data points against one another to compare success rate with a specific variable.

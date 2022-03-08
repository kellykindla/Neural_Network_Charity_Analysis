# Module 19: Neural Network Models 

# Project Overview 
## Purpose of Module 19 
In this module, we expanded our knowledge of machine learning by discovering how to design, train, evaluate, and export neural networks to use in any scenario, with any data set. We used pro-techniques to prepare our input data and to create deep learning models. We explored the structure of neural networks and how changing a component of the model can alter the algorithms performance. We further tested our knowledge by comparing neural network models to traditional machine learning models. 

## Overview of Assignment
For this assignment, we utilized TensorFlow to create a deep learning, neural network model with the goal of vetting potential donation recipients for the company “AlphabetSoup”. By using the data provided with information for each organization, we were able to create a data driven solution in which we could predict which organizations are worth donating to and which are high risk. After preprocessing our data, we compiled, trailed and evaluated our initial neural network model and then altered the models structure in efforts to optimize the results and achieve an accuracy score above 75% to present more confident results. 

## Resources 
- charity_data.csv 
- Scikit-learn version 1.0.2
- Conda version 4.11.0
- Python version 3.8.8
- Jupyter version Notebook 6.3.0
- Pandas version 1.3.5
- TensorFlow version 2.8.0 

# Results 
## Data Preprocessing 
Through an initial analysis of our dataset we find, 12 columns with over 34,000 rows- a sufficient sample size for our analysis. Since the goal of creating our neural network model was to predict where to make investments, the target variable will be the IS_SUCCESSFUL column as that column tells us if the organization used the investment effectively. We find that the EIN and NAME identifier columns are neither target nor feature columns and can be removed from the dataset prior to analysis. When optimizing the neural network model, I also chose to remove the STATUS column as majority (34294 of 34299) of the organizations were “active status (1)” and this column did not offer much insight. In further efforts to optimize the model, I similarly dropped the CLASSIFICATION column since there were 71 different unique values in the column and this could potentially confuse the results. This leaves APPLICATION_TYPE, AFFILIATION, USE_CASE, ORGANIZATION, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT as our feature columns for our model. 
After determining the relevance of each variable to the model, we dropped necessary columns, binned columns were necessary (based on the respective counts for the occurrence of variables for the columns) and created a OneHotEncoder instance to transform the categorical variables to numeric values. We then merged the numeric columns to the original dataset, dropping the original columns, giving us 44 columns before optimization and 38 columns with the optimized neural network model to analyze. In order to prepare our data for the machine learning model, we split our preprocessed data into training and testing data and created a StandardScaler instance to scale the training and testing datasets. 

## Compiling, Training, and Evaluating the Model
To perform the analysis, we created a binary classification model using TensorFlow’s Keras sequential model that can predict if an organization will use the company’s money effectively (is successful or not). The following information describes the neurons, layers, and the activation functions used for the neural network models created for this analysis. 

### Original Model 
For the original model, I followed the guidelines provided in the starter code which displayed two hidden layers- the first layer with 80 neurons and the second with 30 neurons. I set the input features to be equal to the length of the trained dataset as we want there to be an input neuron for each feature of the dataset. For the original neural network, I set each hidden layers activation function to “RELU” as it is efficient and tends to have good convergence performance; however, to best predict the probability of an organization being “successful”, I used a “Sigmoid” activation function for the output layer so results will exist between 0 and 1. The summary structure of the original neural network is shown below with 5,981 parameters. 

<img width="587" alt="og_ss" src="https://user-images.githubusercontent.com/92558842/157148460-43899f4d-b8b3-460d-a089-33e70a696604.png">

This model was then compiled and trained with binary cross entropy and 100 epochs. The model was evaluated for loss and accuracy to find 71.53% loss and 72.70% accuracy. 

<img width="670" alt="og_loss_ac" src="https://user-images.githubusercontent.com/92558842/157148481-8ab529c7-c1f6-43ea-9808-6641bb9d4f92.png">


### First Attempt 
In my first attempt to optimize the model to achieve a higher predictive accuracy, I added another hidden layer while maintaining the same value of input features where it was equal to the length of the trained dataset. The first hidden layer had 80 neurons, the second layer had 40 neurons, and the third 20 with each layer having the activation function set to “RELU”. I did attempt to change the activation function of the hidden layers to both “Sigmoid” and “Tanh” but found that this decreased accuracy and was not worth saving as an attempt to optimize results. The output layer was kept as “sigmoid” for this attempt to best predict results. The summary structure of my first attempt of optimizing the neural network model is shown below. 

<img width="575" alt="first_ss" src="https://user-images.githubusercontent.com/92558842/157148519-0e8d9fe7-0a7a-4081-8330-a0b45b78844c.png">

From the structure we find that the parameters increased to 7,121 which generally indicated better estimates but can also lead to overfitting. This model was further compiled and trained with 100 epochs and was evaulated to find 57.68% loss and 71.94% accuracy. 

<img width="482" alt="first_loss_ac" src="https://user-images.githubusercontent.com/92558842/157148539-cbce5e9f-5c5f-4cc2-acb5-c86b9940a614.png">


### Second Attempt 
In the next attempt to optimize the neural network, I maintained the number of input features and the number of hidden layers. I did increase the number of neurons in each layer where the first hidden layer had 100 neurons, the second 60, and the third 20. I kept the hidden layers activation functions as “RELU” as this provided the best results. I did, however, change the activation function for the output layer to “Tanh”. The summary structure of this attempt is displayed in the image below. 

<img width="571" alt="second_ss" src="https://user-images.githubusercontent.com/92558842/157148566-a2e66de0-bb6a-41ea-8748-dec6ae630782.png">

The structure shows that the parameters further increased to 11,101. This model was compiled and trained with 80 epochs in efforts to optimize performance. The model was evaluated to find 58.76% loss and 72.15% accuracy. 

<img width="470" alt="second_loss_ac" src="https://user-images.githubusercontent.com/92558842/157148602-af0d74f1-26cf-471c-a9a0-dab368ed5d61.png">


### Third Attempt 
In my final attempt to optimize the neural network, I maintained the input features but increased to number of hidden layers and the number of neurons for each layer. The first layer had 160 neurons, the second 100, third 80, fourth 60, and the fifth 20 neurons. The activation function for each hidden layer was “RELU” in attempts to provide the best results. The output activation function was kept as “Tanh” as it performed better than having “Sigmoid” set as the activation function. The summary structure for this attempt is displayed below. 

<img width="589" alt="third_ss" src="https://user-images.githubusercontent.com/92558842/157148617-ffd26bd5-55d6-47ed-8b62-f9c07217c416.png">

The structure shows that the parameters increased to 36,361. This model was then compiled and trailed with 120 epochs and was evaluated to find 76.16% loss and 72.13% accuracy. 

<img width="471" alt="third_loss_ac" src="https://user-images.githubusercontent.com/92558842/157148649-de8fb726-7b3f-413a-b110-a276fe99fbad.png">


### Results 
Despite my best efforts to optimize the model by dropping more columns, adding more neurons to hidden layers, adding more hidden layers, using different activation functions, and adding and reducing the number of epochs, I could not achieve an accuracy score above the original neural network model of 72.70% and thus did not achieve the desired results of 75% accuracy. 

# Summary 
In this challenge we discovered the advantages and limitation of implementing neural networks. For my analysis, I found that altering the structure of the neural network — by adding hidden layers and neurons— does not always guarantee better model performance. For my deep learning model, I found that my original and most simple neural network model performed the best with an accuracy of 72.70%. However, we find that for each attempt to optimize the model, the loss is high— lowest being in the second attempt at 57.68%— meaning that the model generally does not do a good job at predicting if the organization effectively used the company’s investment. With our accuracy generally being high at an average of 72.23% and loss being high, we can assume that our model makes large error in some of the data and is not the best at predicting if the organization used the money effectively (is successful or not). In an attempt to better predict results, this dataset could be analyzed with a logistic regression model since it would predict the probability of an organization being successful or not being successful and the process is much simpler and takes less time and energy. 

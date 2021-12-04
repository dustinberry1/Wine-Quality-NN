#1. Load the dataset wine.csv into memory.
library(readr)
wine <- read.csv("Maryville/Predictive Modeling/wine.csv")
View(wine)

#2. Preprocess the inputs
#2a. Standardize the inputs using the scale() function.
scaled.wine =  scale(wine)

#2b. Convert the standardized inputs to a data frame using the as.data.frame()
#function.
scaled.wine = as.data.frame(scaled.wine)

#2c. Split the data into a training set containing 3/4 of the original data
#(test set containing the remaining 1/4 of the original data).
set.seed(1) #only needed for reproducability
index <- sample(1:nrow(scaled.wine),0.75*nrow(scaled.wine))
train <- scaled.wine[index,]
test <- scaled.wine[-index,]


#3. Build a neural networks model
library(neuralnet)

#3a. The response is quality and the inputs are: volatile.acidity, density, pH,
#and alcohol. Please use 1 hidden layer with 1 neuron.
nn.model <- neuralnet(quality ~ volatile.acidity + density + pH + alcohol, data=train, hidden=c(1))

#3b. Plot the neural networks.
plot(nn.model)

#3c. Forecast the wine quality in the test dataset.
predict.nn = compute(nn.model, test[, c("volatile.acidity", "density", "pH", "alcohol")])

#3d. Get the observed wine quality of the test dataset.
observ.test = test$quality

#3e. Compute test error (MSE).
#The test error for this model is 0.701339
mean((observ.test- predict.nn$net.result)^2)



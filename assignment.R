
data = read.csv("spam.csv",stringsAsFactors = F,na.strings = "")
sms_raw <- data[,1:2]
sms_raw$Category <- factor(sms_raw$Category)
# Check the structure of the dataset
str(sms_raw)
# Check the number of spam and ham messages
table(sms_raw$Category)
prop.table(table(sms_raw$Category))
data[,c("X","X.1","X.2")] <- NULL
colnames(data) <- c("Category","Message")
head(data)
str(data)
data$Category <- factor(data$Category)
library("ggplot2")
g <- ggplot(data = data,aes(x=Category,fill=Category))
g + geom_bar(stat = "count") + ggtitle("Distribution of Messages")
#install.packages("stringr")
library(stringr)
data$length <- str_length(data$Message)
head(data)
ggplot(data = data,aes(x=length,fill = Category))+geom_histogram(binwidth = 5) 
#install.packages("tm")
library(tm)
data_corpus <- VCorpus(x = VectorSource(data$Message))
data_corpus
corpus_clean <- data_corpus

# Remove Numbers
corpus_clean <- tm_map(x = corpus_clean, FUN = removeNumbers)

# Transform all letters to lower case
corpus_clean <- tm_map(x = corpus_clean, content_transformer(tolower))

# Remove punctuation
corpus_clean <- tm_map(x = corpus_clean, FUN = removePunctuation)
# Remove stop words
corpus_clean <- tm_map(x = corpus_clean, FUN = removeWords, stopwords())

#install.packages("SnowballC")

# Load package
library(SnowballC)

# Test the function
wordStem(words = c("cooks", "cooking", "cooked"))
# Stem words in corpus
corpus_clean <- tm_map(x = corpus_clean, FUN = stemDocument)
# Remove extra white spaces
corpus_clean <- tm_map(x = corpus_clean, FUN = stripWhitespace)

# Create Document Term Matrix
DTM <- DocumentTermMatrix(x = corpus_clean)

DTM
# Create Training Set
DTM_train <- DTM[1:round(nrow(DTM)*0.80, 0), ]

# Create Test Set
DTM_test <- DTM[(round(nrow(DTM)*0.80, 0)+1):nrow(DTM), ]

# Create vectors with labels for the training and test set
train_labels <- sms_raw[1:round(nrow(sms_raw)*0.80, 0), ]$Category
test_labels <- sms_raw[(round(nrow(sms_raw)*0.80, 0)+1):nrow(DTM), ]$Category

# Check proportion of ham and spam is similar on the training and test set
prop.table(table(train_labels))

prop.table(table(test_labels))


#install.packages("wordcloud")

# Load package
library(wordcloud)

# Create wordcloud for the whole dataset
wordcloud(words = corpus_clean, 
        min.freq = 100, # minimum number of times a word must be present before it appears 
        random.order = FALSE, # Arrange most frequent words to be in the center of the word cloud
        color = (colors = c("#4575b4","#74add1","#abd9e9","#e0f3f8","#fee090","#fdae61","#f46d43","#d73027"))) 
        

threshold <- 0.1 # in %
min_freq = round(DTM$nrow*(threshold/100),0) # calculate minimum frequency
min_freq

# Create vector of most frequent words
frequent_words <- findFreqTerms(x = DTM, lowfreq = min_freq)

str(frequent_words)
# Filter DTM to only have most frequent words
DTM_train_most_frequent <- DTM_train[, frequent_words]
DTM_test_most_frequent <- DTM_test[, frequent_words]

# Check dimension of DTM
dim(DTM_train_most_frequent)

# Create function  that converts numeric values to "Yes" or "No" if word is present or absent in document
is_present <- function(x) {
  x <- ifelse(test = x > 0, yes = "Yes", no = "No")
}

# Test function
x <- is_present(c(1, 0, 3, 4, 0, 0))
x

# Apply is_present() function to training and test DTM
DTM_train_most_frequent <- apply(X = DTM_train_most_frequent, 
                                 MARGIN = 2, # Apply function to columns
                                 FUN = is_present) # Specify function to be used

DTM_test_most_frequent <- apply(X = DTM_test_most_frequent, 
                                MARGIN = 2, # Apply function to columns
                                FUN = is_present) # Specify function to be used

# Install package
install.packages("e1071")

# Load package
library(e1071)

# Create model from the training dataset
spam_classifier <- naiveBayes(x = DTM_train_most_frequent, y = train_labels)

# Print probability tables for some words
spam_classifier$tables$call

spam_classifier$tables$friend
spam_classifier$tables$free

test_predictions <- predict(object = spam_classifier, newdata = DTM_test_most_frequent)

## Create confusion matrix

# install package
install.packages("caret")

# load caret package
library(caret)

# Create confusion matrix
confusionMatrix(data = test_predictions, reference = test_labels, positive = "spam", dnn = c("Prediction", "Actual"))

# Install package
install.packages("e1071")

# Load package
library(e1071)

# Create model from the training dataset
spam_classifier_with_LE <- naiveBayes(x = DTM_train_most_frequent, 
                                      y = train_labels, 
                                      laplace = 1 # set laplace estimator to 1
)
## Make predictions on test set
test_predictions_with_LE <- predict(object = spam_classifier_with_LE, newdata = DTM_test_most_frequent)

## Create confusion matrix

# install package
 install.packages("caret")

# load caret package
library(caret)

# Create confusion matrix
confusionMatrix(data = test_predictions_with_LE, reference = test_labels, positive = "spam", dnn = c("Prediction", "Actual"))

CM <- confusionMatrix(data = test_predictions, reference = test_labels); 
naive_Accuracy <- round(CM$overall[["Accuracy"]], 4)*100

CM_LE <- confusionMatrix(data = test_predictions_with_LE, reference = test_labels); 
naive_LE_Accuracy <- round(CM_LE$overall[["Accuracy"]], 4)*100

data.frame(`Without Laplace Estimator` = naive_Accuracy, 
           `With Laplace Estimator` = naive_LE_Accuracy, 
           row.names = c("Accuracy"))

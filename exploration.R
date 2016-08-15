# install.packages("ggplot2")
# install.packages("wordcloud")

library(ggplot2)
library(wordcloud)

data <- read.csv("C:/BigData/TextMining/hispatweets/res2.txt", encoding="UTF-8", quote="", sep = "\t", check.names = F)

head(data[,1:20])
head(data[,2017:2018])

female <- data[data$sex == "female",]
male <- data[data$sex == "male",]
unknown <- data[data$sex == "UNKNOWN",]
head(female)
head(male)
head(unknown)


plot(data[,1], data[,2], xlab = colnames(data)[1], ylab = colnames(data)[2])


bysex <- by(data[,1:2000], data$sex, colMeans)

plot(bysex$male, bysex$female)
identify(bysex$male, bysex$female)


boxplot(data)
boxplot(bysex)


malesort <- sort(bysex$male, decreasing = T)
femalesort <- sort(bysex$female, decreasing = T)
unknownsort <- sort(bysex$UNKNOWN, decreasing = T)
malesort[1:20]
femalesort[1:20]
unknownsort[1:20]


head(data[,2001:2017])
str(data[,2001:2017])
bysex_extra <- by(data[,2001:2017], data$sex, colMeans)
plot(factor(data$sex), data$positive)
plot(factor(data$sex), data$negative)
plot(factor(data$sex), data$positive + data$negative)
plot(data$positive, data$negative)

p <- ggplot(data, aes(factor(data$sex), data$positive))
p <- p + geom_point(aes())
p


ggplot(data, aes(factor(data$sex), positive)) + 
  geom_bar(stat = "identity") 
ggplot(data, aes(factor(data$sex), negative)) + 
  geom_bar(stat = "identity") 

  
plot(factor(data$sex), data$number.of.tokens)
plot(factor(data$sex), data$mean.size.of.tokens)
plot(factor(data$sex), data$maximum.word)
plot(factor(data$sex), data$sentence.size.mean)
plot(factor(data$sex), data$followers.count)
plot(factor(data$sex), data$friends.count)
plot(factor(data$sex), data$favourites.count)
plot(factor(data$sex), data$statuses.count)
plot(factor(data$sex), data$number.of.tokens)

p <- ggplot(data, aes(data$sex, data$positive+data$negative))
p <- p + geom_boxplot()
p

head(data[,2001:2017])
p <- ggplot(data, aes(data$sex, data[,2007]))
p <- p + geom_boxplot() + ylab("Tamaño medio de frases") + xlab("Sexo")
p


# Colors
head(data[,2013:2017])
str(data[,2013:2017])

as.hexmode(8045550)

ggplot(data, aes(data$sex, profile_sidebar_border_color)) + 
  geom_point(aes(color = profile_sidebar_border_color)) 
ggplot(data, aes(data$sex, profile_background_color)) + 
  geom_point(aes(color = profile_background_color)) 
ggplot(data, aes(data$sex, profile_link_color)) + 
  geom_point(aes(color = profile_link_color)) 
ggplot(data, aes(data$sex, profile_text_color)) + 
  geom_point(aes(color = profile_text_color)) 
ggplot(data, aes(data$sex, profile_sidebar_fill_color)) + 
  geom_point(aes(color = profile_sidebar_fill_color)) 


bycountry <- by(data[,1:2000], data$country, colMeans)
argentinasort <- sort(bycountry$argentina, decreasing = T)
chilesort <- sort(bycountry$chile, decreasing = T)
colombiasort <- sort(bycountry$colombia, decreasing = T)
espanasort <- sort(bycountry$espana, decreasing = T)
mexicosort <- sort(bycountry$mexico, decreasing = T)
perusort <- sort(bycountry$peru, decreasing = T)
venezuelasort <- sort(bycountry$venezuela, decreasing = T)
argentinasort[1:20]
chilesort[1:20]
colombiasort[1:20]
espanasort[1:20]
mexicosort[1:20]
perusort[1:20]
venezuelasort[1:20]


freq <- colMeans(data[,1:2000])
freqMale <- colMeans(male[,1:2000])
freqFemale <- colMeans(female[,1:2000])
freqUnknown <- colMeans(unknown[,1:2000])

wf <- data.frame(word=names(freq), freq=freq)  

# p<-ggplot(diamonds, aes(clarity, group = color, colour=color))
# p+geom_line(aes(y=..count..), stat="count")

p <- ggplot(subset(wf, freq>0.05), aes(word, freq))
p <- p + geom_bar(stat="identity")
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))
p

set.seed(142)
wordcloud(names(freq), freq, max.words=40)

wordcloud(names(freq), freq, min.freq=0.05)
wordcloud(names(freqMale), freqMale, min.freq=0.05, colors=brewer.pal(6, "Dark2"))
wordcloud(names(freqFemale), freqFemale, min.freq=0.05, colors=brewer.pal(6, "Dark2"))
wordcloud(names(freqUnknown), freqUnknown, min.freq=0.05, colors=brewer.pal(6, "Dark2"))

wordcloud(names(freq), freq, max.words=40)

wordcloud(names(freq), freq, max.words=20, colors=brewer.pal(6, "Dark2"))
wordcloud(names(freqMale), freq, max.words=50, colors=brewer.pal(6, "Dark2"))
wordcloud(names(freqFemale), freq, max.words=50, colors=brewer.pal(6, "Dark2"))
wordcloud(names(freqUnknown), freq, max.words=50, colors=brewer.pal(6, "Dark2"))

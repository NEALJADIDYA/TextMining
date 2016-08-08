data <- read.delim("C:/BigData/TextMining/hispatweets/res.txt", encoding="UTF-8", quote="")

head(data[,c(2017:2018)])

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


head(data[,2001:2016])
bysex_extra <- by(data[,2001:2016], data$sex, colMeans)
plot(factor(data$sex), data$positive)
plot(factor(data$sex), data$negative)
plot(factor(data$sex), data$positive + data$negative)
plot(data$positive, data$negative)

plot(factor(data$sex), data$number.of.tokens)
plot(factor(data$sex), data$mean.size.of.tokens)
plot(factor(data$sex), data$maximum.word)
plot(factor(data$sex), data$sentence.size.mean)
plot(factor(data$sex), data$followers.count)
plot(factor(data$sex), data$friends.count)
plot(factor(data$sex), data$favourites.count)
plot(factor(data$sex), data$statuses.count)
plot(factor(data$sex), data$number.of.tokens)


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

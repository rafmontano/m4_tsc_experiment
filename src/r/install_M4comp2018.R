#https://github.com/carlanetto/M4comp2018

install.packages("https://github.com/carlanetto/M4comp2018/releases/download/0.2.0/M4comp2018_0.2.0.tar.gz",
                 repos=NULL, force=TRUE)

library(M4comp2018)
data(M4)
names(M4[[1]])

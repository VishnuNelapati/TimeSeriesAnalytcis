library("forecast")
library("zoo")
library('ggplot2')

#-----------------------------------Get Data------------------------------------
retail <- read.csv(file.choose())
head(retail)

tail(retail)

sales.ts <- ts(retail$Sales,start = c(1992,1),frequency = 12)
head(sales.ts)
tail(sales.ts)

#-----------------------------Visualize Data ----------------------------------

plot(sales.ts,xlim=c(1992,2022),ylim=c(1500,7500),
     bty="l",main= "Reatil Sales : Beer,Liquor and Wine Stores",
     lwd=2,ylab = "Sales (In $Millions)")
axis(1, at = seq(1992, 2022, 1), labels = format(seq(1992, 2022, 1)) )
points(sales.ts,col=rainbow(12),pch=19,cex=1)
legend(1992,7500,legend = c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"),
       col=rainbow(12),pch =19,title = "Months",cex = 1,ncol = 4)

sales.stl <- stl(sales.ts,s.window = 'periodic')
autoplot(sales.stl,main="Time Series components of Retail Sales")

boxplot(sales.ts ~ cycle(sales.ts),ylab = "Sales (In $Millions)",xlab = "Months",
        main = "Boxplot of Retail Sales")


Acf(sales.ts,lag.max = 12,main="AutoCorrelation Plot For Retail Sales")

#-------------------------------Partitioning Data ------------------------------

#spliting to training and validation 

nvalid <- 70
ntrain <- length(sales.ts) - nvalid
train.ts <- window(sales.ts,start = c(1992,1),end = c(1992,ntrain))
valid.ts <- window(sales.ts,start = c(1992,ntrain+1),end = c(1992,ntrain+nvalid))

#-----------------------------Apply and compare---------------------------------

#----------Predictability test -------
summary(Arima(sales.ts,order = c(1,0,0)))

pnorm((0.8638 - 1)/(0.0274))

Acf(diff(sales.ts,lag = 1),lag.max = 12,main = "Autocorrelation Plot for First difference retail Sales")

#----------------------------Linear Trend and seasonality ----------------------

train.lin <- tslm(train.ts ~ trend + season)
summary(train.lin)

train.lin.pred <- forecast(train.lin,h=nvalid,level = 0)
train.lin.pred$mean

plot(sales.ts, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(1000,8500), bty = "l",
     xlim = c(1992, 2024), main = "Linear Regression Model with Trend and Seasonality \n Training and validation Partitions"
     ,col='blue',lwd=1) 
axis(1, at = seq(1992, 2024, 1), labels = format(seq(1992, 2024, 1)) )
lines(train.lin$fitted, col = "#D35400", lwd = 2, lty = 1)
lines(train.lin.pred$mean, col = "#D35400", lwd = 2, lty = 2)

legend(1992,7000, legend = c("Retail Sales Data", 
                             "Regression Forecast training partition", 
                             "Regression Forecast Validation Partition"),
       col = c("blue", "#D35400", "#D35400"), 
       lty = c(1,1,2),lwd =c(1,2,2), bty = "n")

lines(c(2015.5, 2015.5), c(1000,8000))
lines(c(2021.33, 2021.33), c(1000,8000))
text(2003.25, 7750, "Training")
text(2018.25, 7750, "Validation")
text(2023.25, 7750, "Future")
arrows(2015.25,7500, 1991.75,7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015.5,7500, 2021.1, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 7500, 2025, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#Residuals
lin.train.residuals <- train.lin$residuals
lin.train.residuals
Acf(lin.train.residuals,lag.max = 24,
    main="AutoCorrelation for residuals of Linear Regression Trend and Seasona model")


plot(lin.train.residuals,col="#DE3163",main="Residuals of Linear Regression Trend and Seasona model",
     ylab="Residuals",bty="l",ylim=c(-600,600),xlim=c(1992,2022))
lines(c(1992,2022),c(0,0),lty=2)
legend(1992,650,legend = c("Linear Regression trend & Seasonal model Residuals Training Partition",
                           "Zero Reference Line"),
       col=c("#DE3163","Black"),lty=c(1,2),lwd=c(2,1),bty="n")


lin.valid.residuals <- valid.ts - train.lin.pred$mean
lin.valid.residuals

#trailing ma model for regression residuals - training/validation partition 

ma.trailing.res.8 <- rollmean(lin.train.residuals,k=8,align = 'right')
#ma.trailing.res.8

ma.trailing.res.8.pred <- forecast(ma.trailing.res.8,h=nvalid,level = 0)
ma.trailing.res.8.pred$mean

plot(lin.train.residuals,col="#DE3163",main="Trailing MA for regression residuals",
     ylab="Residuals",bty="l",ylim=c(-500,2000),xlim=c(1992,2024))
lines(lin.valid.residuals,col="#DE3163",lty=1)
lines(ma.trailing.res.8.pred$mean,col="blue",lwd=2,lty=2)
lines(ma.trailing.res.8,col="blue",lwd=2)
lines(c(1992,2024),c(0,0),lty=2)
legend(1992,1800,legend = c("Linear Regression trend & Seasonal model Residuals",
                           "Trailing MA forecast for regression residuals - Training Partition",
                           "Trailing MA forecast for Regression residuals - validation Partition",
                           "Zero Reference Line"),
       col=c("#DE3163","blue","blue","Black"),lty=c(1,1,2,2),lwd=c(1,2,2,2),bty="n")
lines(c(2015.5, 2015.5), c(-600,2000))
lines(c(2021.33, 2021.33), c(-600,2000))
text(2003.25, 2000, "Training")
text(2018.25, 2000, "Validation")
text(2023.25, 2000, "Future")
arrows(2015.25,1850, 1991.75,1850, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015.5,1850, 2021.1, 1850, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 1850, 2025, 1850, code = 3, length = 0.1,
       lwd = 1, angle = 30)


#two level forecast for training/validation partition

two.level.lin <- ma.trailing.res.8.pred$mean + train.lin.pred$mean
two.level.lin 


valid.df <- data.frame(valid.ts, train.lin.pred$mean, 
                       ma.trailing.res.8.pred$mean, 
                       two.level.lin )
names(valid.df) <- c("Sales", "Regression Forecast", 
                     "Trailing MA Residuals", "Combines forecast")
valid.df


round(accuracy(train.lin.pred$mean,valid.ts),3)
round(accuracy(two.level.lin ,valid.ts),3)

#total Data 

total.lin <- tslm(sales.ts ~ trend  + season)
summary(total.lin)

total.lin.pred <- forecast(total.lin,h=24,level = 0)
total.lin.pred


total.trail.ma <- rollmean(total.lin$residuals,k=8,align = 'right')
total.trail.ma

total.trail.ma.pred <- forecast(total.trail.ma,h=24,level = 0)
total.trail.ma.pred

total.two.level.lin  <- total.lin.pred$mean + total.trail.ma.pred$mean
total.two.level.lin 

#Cretaing a combines forecast dataframe
forecast.df <- data.frame(total.lin.pred$mean,total.trail.ma.pred$mean,total.two.level.lin )

colnames(forecast.df) <- c("Regression Forecast","Trailing MA residual Forecast","Total Forecast")

forecast.df

#Acucuracy
round(accuracy(total.lin.pred$fitted,sales.ts),3) # only linear model
round(accuracy(total.lin.pred$fitted+total.trail.ma,sales.ts),3) # two level forecast with moving average
round(accuracy(naive(sales.ts)$fitted,sales.ts),3) #naive
round(accuracy(snaive(sales.ts)$fitted,sales.ts),3) # snaive

#Linear Regression with trend & Season and Trailing MA for residuals

plot(sales.ts, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(1000,8500), bty = "l",
     xlim = c(1992, 2024), main = "Two-Level Forecast For entire Data \n Linear Regression with trend & Season and Trailing MA for residuals"
     ,col='blue',lwd=1) 
axis(1, at = seq(1992, 2024, 1), labels = format(seq(1992, 2024, 1)) )
lines(total.two.level.lin , col = "#D35400", lwd = 2, lty = 2)

legend(1992,7000, legend = c("Retail Sales Data", 
                             "Two-level Forecast For Future 24 periods"),
       col = c("blue", "#D35400"), 
       lty = c(1,2),lwd =c(1,2), bty = "n")

lines(c(2021.33, 2021.33), c(1000,8000))
text(2008.25, 7750, "Training")
text(2023.25, 7750, "Future")
arrows(2021.25,7500, 1991.75,7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 7500, 2025, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#-------------------------------Quadratic Model---------------------------------

train.quad <- tslm(train.ts ~ trend + I(trend^2)+season)
summary(train.quad)

train.quad.pred <- forecast(train.quad,h=nvalid,level = 0)
train.quad.pred$mean


plot(sales.ts, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(1000,8500), bty = "l",
     xlim = c(1992, 2024), main = "Quadratic Trend and seaonality Forecast in Training and validation Partitions"
     ,col='blue',lwd=1) 
axis(1, at = seq(1992, 2024, 1), labels = format(seq(1992, 2024, 1)) )
lines(train.quad$fitted, col = "#D35400", lwd = 2, lty = 1)
lines(train.quad.pred$mean, col = "#D35400", lwd = 2, lty = 2)

legend(1992,7000, legend = c("Retail Sales Data", 
                             "Quadratic Trend and seaonality Forecast training partition", 
                             "Quadratic Trend and seaonality Forecast Validation Partition"),
       col = c("blue", "#D35400", "#D35400"), 
       lty = c(1,1,2),lwd =c(1,2,2), bty = "n")

lines(c(2015.5, 2015.5), c(1000,8000))
lines(c(2021.33, 2021.33), c(1000,8000))
text(2003.25, 7750, "Training")
text(2018.25, 7750, "Validation")
text(2023.25, 7750, "Future")
arrows(2015.25,7500, 1991.75,7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015.5,7500, 2021.1, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 7500, 2025, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#trailing ma model for Quadratic Trend and seasonality residuals
Quad.train.residuals <- train.quad$residuals
Quad.train.residuals
Acf(Quad.train.residuals,lag.max = 24,
    main="AutoCorrelation for residuals of Linear Regression Trend and Seasona model")

#Moving average for quadratic model residuals
ma.trailing.quadres.8 <- rollmean(Quad.train.residuals,k=8,align = 'right')
ma.trailing.quadres.8

ma.trailing.quadres.8.pred <- forecast(ma.trailing.quadres.8,h=nvalid,level = 0)
ma.trailing.quadres.8.pred$mean

Quad.valid.residuals <- valid.ts - train.quad.pred$mean
Quad.valid.residuals

plot(Quad.train.residuals,col="#DE3163",main="Trailing MA for Quadratic regression residuals",
     ylab="Residuals",bty="l",ylim=c(-500,2000),xlim=c(1992,2024))
lines(Quad.valid.residuals,col="#DE3163",lty=1)
lines(ma.trailing.quadres.8.pred$mean,col="blue",lwd=2,lty=2)
lines(ma.trailing.quadres.8,col="blue",lwd=2)
lines(c(1992,2024),c(0,0),lty=2)
legend(1992,1800,legend = c("Quadratic trend & Seasonal model Residuals",
                            "Trailing MA forecast for regression residuals - Training Partition",
                            "Trailing MA forecast for Regression residuals - validation Partition",
                            "Zero Reference Line"),
       col=c("#DE3163","blue","blue","Black"),lty=c(1,1,2,2),lwd=c(1,2,2,2),bty="n")
lines(c(2015.5, 2015.5), c(-600,2000))
lines(c(2021.33, 2021.33), c(-600,2000))
text(2003.25, 2000, "Training")
text(2018.25, 2000, "Validation")
text(2023.25, 2000, "Future")
arrows(2015.25,1850, 1991.75,1850, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015.5,1850, 2021.1, 1850, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 1850, 2025, 1850, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#two level forecast

two.level.quad <- ma.trailing.quadres.8.pred$mean + train.quad.pred$mean
two.level.quad


valid.df <- data.frame(valid.ts, train.quad.pred$mean, 
                       ma.trailing.quadres.8.pred$mean, 
                       two.level.quad)
names(valid.df) <- c("Sales", "Qudratic Trend and seaonality Forecast", 
                     "Trailing MA Residuals", "Combines forecast")
valid.df


round(accuracy(train.quad.pred$mean,valid.ts),3)
round(accuracy(two.level.quad,valid.ts),3)

#total Data 

total.Quad <- tslm(sales.ts ~ trend  +I(trend^2)+ season)
summary(total.Quad)

total.Quad.pred <- forecast(total.Quad,h=24,level = 0)
total.Quad.pred


total.quad.trail.ma <- rollmean(total.Quad$residuals,k=8,align = 'right')
total.quad.trail.ma

total.quad.trail.ma.pred <- forecast(total.quad.trail.ma,h=24,level = 0)
total.quad.trail.ma.pred

total.two.level.quad <- total.Quad.pred$mean + total.quad.trail.ma.pred$mean
total.two.level.quad

forecast.df <- data.frame(total.Quad.pred$mean,total.quad.trail.ma.pred$mean,total.two.level.quad)

colnames(forecast.df) <- c("Qudratic Trend and seaonality Forecast","Trailing MA residual Forecast","Total Forecast")

forecast.df


round(accuracy(total.Quad.pred$fitted,sales.ts),3) # only quadratic model
round(accuracy(total.Quad.pred$fitted+total.quad.trail.ma,sales.ts),3) # two level forecast
round(accuracy(naive(sales.ts)$fitted,sales.ts),3) #naive
round(accuracy(snaive(sales.ts)$fitted,sales.ts),3) # snaive


plot(sales.ts, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(1000,8500), bty = "l",
     xlim = c(1992, 2024), main = "Two Level Forecast for entire data \n Quadratic Model with trend & Season and Trailing MA for residuals"
     ,col='blue',lwd=1) 
axis(1, at = seq(1992, 2024, 1), labels = format(seq(1992, 2024, 1)) )
lines(total.two.level.quad, col = "#D35400", lwd = 2, lty = 2)

legend(1992,7000, legend = c("Retail Sales Data", 
                             "Two Level Forecast for future 24 periods"),
       col = c("blue", "#D35400"), 
       lty = c(1,2),lwd =c(1,2), bty = "n")

lines(c(2021.33, 2021.33), c(1000,8000))
text(2008.25, 7750, "Training")
text(2023.25, 7750, "Future")
arrows(2021.25,7500, 1991.75,7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 7500, 2025, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#---------------------------------Holt-winters model ----------------------------

hw.optimal.train <- ets(train.ts,model='ZZZ')
summary(hw.optimal.train)

plot(hw.optimal.train)

hw.optimal.train.pred <- forecast(hw.optimal.train,h=nvalid,level = 0)
hw.optimal.train.pred$mean

#Holt-winters residuals:
hw.train.residuals <- train.ts - hw.optimal.train$fitted
Acf(hw.train.residuals,lag.max = 12,main="Auto_correlation plot for Residuals Of Holt-Winter's Optimal Model")

hw.train.residuals.ar12 <- Arima(hw.train.residuals,order = c(12,0,0))
summary(hw.train.residuals.ar12)
hw.train.residuals.ar12.pred <- forecast(hw.train.residuals.ar12,h=nvalid,level = 0)
hw.train.residuals.ar12.pred$mean

plot(hw.train.residuals,bty="l",col="#DE3163",
     main="AR(12) Model for Residuals Of Holt-Winter's Model \n Training Partition and validation Partitions",xlim=c(1992,2025),ylim=c(-200,2000))
lines(c(1991.75,2024),c(0,0),lty=2)
lines(valid.ts-hw.optimal.train.pred$mean,col="#DE3163",lty = 2)
lines(hw.train.residuals.ar12$fitted,col='#48C9B0',lwd=2)
lines(hw.train.residuals.ar12.pred$mean,col='#48C9B0',lwd=2,lty=2)
legend(1992,1500,legend = c("Holt-Winter's Model Residuals-Training Partition",
                            "Holt-Winter's Model Residuals-Validation Partition",
                            "Residuals Forecast using AR(12) model-Training Partition",
                           "Residuals Forecast using AR(12) model-Validation Partition",
                           "Zero Line"),
       col=c("#DE3163","#DE3163","#48C9B0","#48C9B0","black"),
       lty=c(1,2,1,2,2),lwd=c(1,1,2,2,2),bty="n")
lines(c(2015.5, 2015.5), c(-600,2000))
lines(c(2021.33, 2021.33), c(-600,2000))
text(2003.25, 2000, "Training")
text(2018.25, 2000, "Validation")
text(2023.25, 2000, "Future")
arrows(2015.25,1850, 1991.75,1850, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015.5,1850, 2021.1, 1850, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 1850, 2025, 1850, code = 3, length = 0.1,
       lwd = 1, angle = 30)


Acf(hw.train.residuals.ar12$residuals,lag.max = 12,
    main="Auto-Correlation plot for Reisulas of Ar(12) Model residuals")

hw.train.two.level <- hw.train.residuals.ar12.pred$mean + hw.optimal.train.pred$mean
hw.train.two.level

#Holt-Winters Optimal for train
round(accuracy(hw.optimal.train.pred,valid.ts),3)

#Holt-winter + Ar(12) for residuals
round(accuracy(hw.train.two.level,valid.ts),3)

plot(sales.ts, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(1000,8500), bty = "l",
     xlim = c(1992, 2024), main = "Two-Level Model for Training and validation Partitions \n Holt-Winter's Model and AR(12) for Residuals"
     ,col='blue',lwd=1) 
axis(1, at = seq(1992, 2024, 1), labels = format(seq(1992, 2024, 1)) )
lines(hw.optimal.train.pred$fitted + hw.train.residuals.ar12$fitted,col='#D35400',lwd=2)
lines(hw.train.two.level,col='#D35400',lwd=2,lty=2)

legend(1992,7000, legend = c("Retail Sales Data", 
                             "Two-Level Model Forecast - Training partition", 
                             "Two-Level Model Forecast - Validation Partition"),
       col = c("blue", "#D35400", "#D35400"), 
       lty = c(1,1,2),lwd =c(1,2,2), bty = "n")

lines(c(2015.5, 2015.5), c(1000,8000))
lines(c(2021.33, 2021.33), c(1000,8000))
text(2003.25, 7750, "Training")
text(2018.25, 7750, "Validation")
text(2023.25, 7750, "Future")
arrows(2015.25,7500, 1991.75,7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015.5,7500, 2021.1, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 7500, 2025, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#for total data 

hw.optimal.total <- ets(sales.ts,model='ZZZ')
summary(hw.optimal.total)

plot(hw.optimal.total)

hw.optimal.total.pred <- forecast(hw.optimal.total,h=24,level = c(80,95))
hw.optimal.total.pred$mean


#AR(12) and two level model with Holt-winters residuals:
hw.total.residuals <- sales.ts - hw.optimal.total$fitted
plot(hw.total.residuals)

Acf(hw.total.residuals,lag.max = 12,main="AutoCorrelation Plot for residuals - Entire Data")

hw.total.residuals.ar12 <- Arima(hw.total.residuals,order = c(12,0,0))
hw.total.residuals.ar12.pred <- forecast(hw.total.residuals.ar12,h=24,level = 0)
hw.total.residuals.ar12.pred$mean

Acf(hw.total.residuals.ar12$residuals,lag.max = 12,
    main="Auto-Correlation plot for Reisulas of Ar(12) Model residuals")

hw.total.two.level <- hw.total.residuals.ar12.pred$mean + hw.optimal.total.pred$mean
hw.total.two.level

hw.forecast.df <- cbind(hw.optimal.total.pred$mean,hw.total.residuals.ar12.pred$mean,
                        hw.total.two.level)

colnames(hw.forecast.df) <- c("Holt-Winter's Forecast","AR(12) Forecast for residuals",
                              "Combined forecast")

hw.forecast.df

#Holt-Winters Optimal for total data
round(accuracy(hw.optimal.total$fitted,sales.ts),3)

#Holt-winter + Ar(12) for residuals
round(accuracy(hw.optimal.total$fitted+hw.total.residuals.ar12$fitted,sales.ts),3)

plot(sales.ts, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(1000,10500), bty = "l",
     xlim = c(1992, 2024), main = "Two-Level Forecast Model For Future 24 Periods \nHolt-Winter's Optimal Model + AR(12) for residuals"
     ,col='blue',lwd=1) 
axis(1, at = seq(1992, 2024, 1), labels = format(seq(1992, 2024, 1)) )
#lines(hw.optimal.total.pred$fitted,col='#D35400',lwd=2)
lines(hw.total.two.level,col='#D35400',lwd=2,lty=2)

legend(1992,9000, legend = c("Retail Sales Data", 
                             "Two-level Model Forecast Future 24 periods"),
       col = c("blue", "#D35400"), 
       lty = c(1,2),lwd =c(1,2), bty = "n")
lines(c(2021.33, 2021.33), c(1000,9500))
text(2008.25, 9750, "Training")
text(2023.25, 9750, "Future")
arrows(2021.25,9500, 1991.75,9500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 9500, 2025, 9500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#----------------------------Arima Explanation------------------------------
summary(Arima(sales.ts,c(2,0,0))) #AR(2)
summary(Arima(sales.ts,c(0,0,1))) # MA(1)

#Using logarithmic function
a <- par(mfrow = c(2,2))
plot(sales.ts,main="Retail Sales",ylab="Sales",col='blue',xlim=c(1991.75,2021.75),sub="[1,1]",col.sub='red')
lines(c(1992,2022),c(mean(sales.ts),mean(sales.ts)),lty=2)

plot(log(sales.ts),main="Logarithm of Retail Sales",ylab="Adjusted Sales",col='blue',xlim=c(1991.75,2021.75),sub="[1,2]",col.sub='red')
lines(c(1992,2022),c(mean(log(sales.ts)),mean(log(sales.ts))),lty=2)

plot(diff(log(sales.ts),lag = 1),main="First Order Difference",ylab="Adjusted Sales",col='blue',xlim=c(1991.75,2021.75),sub="[2,1]",col.sub='red')
lines(c(1992,2022),c(mean(diff(log(sales.ts),lag = 1)),mean(diff(log(sales.ts),lag = 1))),lty=2)

plot(diff(diff(log(sales.ts),lag = 1),lag = 1),main="Second Order Difference",ylab="Adjusted Sales",col='blue',xlim=c(1991.75,2021.75),sub="[2,2]",col.sub='red')
lines(c(1992,2022),c(mean(diff(diff(log(sales.ts),lag = 1),lag = 1)),mean(diff(diff(log(sales.ts),lag = 1),lag = 1))),lty=2)

par(a)


b <- par(mfrow = c(2,3))
plot(sales.ts,main="Retail Sales",ylab="Sales",col='blue',xlim=c(1991.75,2021.75),sub="[1,1]",col.sub='red')
lines(c(1992,2022),c(mean(sales.ts),mean(sales.ts)),lty=2)

plot(log(sales.ts),main="Logarithm of Retail Sales",ylab="Adjusted Sales",col='blue',xlim=c(1991.75,2021.75),sub="[1,2]",col.sub='red')
lines(c(1992,2022),c(mean(log(sales.ts)),mean(log(sales.ts))),lty=2)

plot(diff(log(sales.ts),lag = 1),main="Only First Order Difference",
     ylab="Adjusted Sales",col='blue',xlim=c(1991.75,2021.75),sub="[1,3]",col.sub='red')
lines(c(1992,2022),c(mean(diff(log(sales.ts),lag = 1)),mean(diff(log(sales.ts),lag = 1))),lty=2)

plot(diff(log(sales.ts),lag = 12),main="Only First Order Seasonal Difference",
     ylab="Adjusted Sales",col='blue',xlim=c(1991.75,2021.75),sub="[2,1]",col.sub='red')
lines(c(1992,2022),c(mean(diff(log(sales.ts),lag = 12)),mean(diff(log(sales.ts),lag = 12))),lty=2)

plot(diff(diff(log(sales.ts),lag = 1),lag = 12),xlim=c(1991.75,2021.75),
     main="First order difference & \n First order Seasonal difference",col='blue',ylab="Adjusted Sales",sub="[2,2]",col.sub='red')
lines(c(1992,2022),c(mean(diff(diff(log(sales.ts),lag = 1),lag = 12)),mean(diff(diff(log(sales.ts),lag = 1),lag = 12))),lty=2)

plot(diff(diff(log(sales.ts),lag = 12),lag = 1),xlim=c(1991.75,2021.75),
     main="First order Seasonal difference & \n First order difference",col='blue',ylab="Adjusted Sales",sub="[2,3]",col.sub='red')
lines(c(1992,2022),c(mean(diff(diff(log(sales.ts),lag = 12),lag = 1)),mean(diff(diff(log(sales.ts),lag = 12),lag = 1))),lty=2)
par(b)

dev.off()

#-------------------------------Auto Arima Train---------------------------------
auto.train <- auto.arima(train.ts)
summary(auto.train)

auto.train.pred <- forecast(auto.train,h = nvalid,level = 0)
auto.train.pred$mean

plot(sales.ts, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(1000,8500), bty = "l",
     xlim = c(1992, 2023.5), main = "Auto Arima Forecast in Training and validation Partitions"
     ,col='blue',lwd=1) 
axis(1, at = seq(1992, 2024, 1), labels = format(seq(1992, 2024, 1)) )
lines(auto.train$fitted,col='#D35400',lwd=2)
lines(auto.train.pred$mean,col='#D35400',lwd=2,lty=2)

legend(1992,7000, legend = c("Retail Sales Data", 
                             "Auto ARIMA(3,1,2)(0,1,2)[12] forecast - Training partition", 
                             "Auto ARIMA(3,1,2)(0,1,2)[12] Forecast -  Validation Partition"),
       col = c("blue", "#D35400", "#D35400"), 
       lty = c(1,1,2),lwd =c(1,2,2), bty = "n")

lines(c(2015.5, 2015.5), c(1000,8000))
lines(c(2021.33, 2021.33), c(1000,8000))
text(2003.25, 7750, "Training")
text(2018.25, 7750, "Validation")
text(2022.75, 7750, "Future")
arrows(2015.25,7500, 1991.75,7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015.5,7500, 2021.1, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 7500, 2024, 7500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

round(accuracy(auto.train.pred,valid.ts),3)

#-------------------------------(3,1,2)(0,1,2)[12] total data ------------------------------

arima.total <- Arima(sales.ts,order = c(3,1,2),seasonal = c(0,1,2))
summary(arima.total)

arima.total.pred <- forecast(arima.total,h = 24,level = c(80,95))
arima.total.pred$mean

plot(sales.ts, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(1000,10500), bty = "l",
     xlim = c(1992, 2025), main = "ARIMA(3,1,2)(0,1,2)[12] forecast for entire data"
     ,col='blue',lwd=2) 
axis(1, at = seq(1992, 2025, 2), labels = format(seq(1992, 2025, 2)) )
#lines(arima.total$fitted,col='#D35400',lwd=2)
lines(arima.total.pred$mean,col='#D35400',lwd=2,lty=2)

legend(1992,8000, legend = c("Retail Sales Data", 
                             "ARIMA(3,1,2)(0,1,2)[12] Forecast for future 24 periods"),
       col = c("blue", "#D35400"), 
       lty = c(1,2),lwd =c(1,2), bty = "n")

lines(c(2021.33, 2021.33), c(1000,10000))
text(2008.25, 9750, "Training")
text(2023.25, 9750, "Future")
arrows(2021.25,9500, 1991.75,9500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 9500, 2025, 9500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#with confidence intervals 

plot(arima.total.pred, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(3000,9500), bty = "l",
     xlim = c(2018, 2024), main = "80% and 95% Confidence Intervals of ARIMA(3,1,2)(0,1,2)[12] Forecast \n For future 24 periods"
     ,col='blue',lwd=2) 
axis(1, at = seq(2018, 2024, 1), labels = format(seq(2018, 2024, 1)) )
#lines(arima.total$fitted,col='#D35400',lwd=2)
#lines(sales.ts,col='#blue',lwd=2,lty=1)

legend(2018.25,9500, legend = c("Retail Sales Data", "Point Forecast for Future 24 periods",
                             "95% Confidence Interval", 
                             "80% Confidence Interval"),
       col = c("blue","#469ee0", "#dbdbdf", "#b1b5ce"), 
       lty = c(1,1,1,1),lwd =c(2,2,5,5), bty = "n")

lines(c(2021.33, 2021.33), c(1000,10000))
text(2022.5, 9550, "Future 24 Periods")
arrows(2021.5, 9250, 2023.5, 9250, code = 3, length = 0.1,
       lwd = 1, angle = 30)

round(accuracy(arima.total$fitted,sales.ts),3)

#--------------------------------auto arima for entire data ------------------------

auto.total <- auto.arima(sales.ts)
summary(auto.total)

auto.total.pred <- forecast(auto.total,h = 24,level = c(80,95))
auto.total.pred$mean

plot(sales.ts, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(1000,10500), bty = "l",
     xlim = c(1992, 2025), main = "Auto ARIMA(2,1,1)(0,1,2)[12] forecast Using entire data"
     ,col='blue',lwd=2) 
axis(1, at = seq(1992, 2025, 2), labels = format(seq(1992, 2025, 2)) )
#lines(arima.total$fitted,col='#D35400',lwd=2)
lines(auto.total.pred$mean,col='#D35400',lwd=2,lty=2)

legend(1992,8000, legend = c("Retail Sales Data", 
                             "Auto ARIMA(2,1,1)(0,1,2)[12] Forecast for future 24 periods"),
       col = c("blue", "#D35400"), 
       lty = c(1,2),lwd =c(1,2), bty = "n")

lines(c(2021.33, 2021.33), c(1000,10000))
text(2008.25, 9750, "Training")
text(2023.25, 9750, "Future")
arrows(2021.25,9500, 1991.75,9500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 9500, 2025, 9500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#Confidence interval

plot(auto.total.pred, 
     xlab = "Time", ylab = "sales (in $millions)", ylim = c(3000,9500), bty = "l",
     xlim = c(2018, 2024), main = "80% and 95% Confidence Intervals of Auto ARIMA(2,1,1)(0,1,2)[12] \nForecast For future 24 periods"
     ,col='blue',lwd=2) 
axis(1, at = seq(2018, 2024, 1), labels = format(seq(2018, 2024, 1)) )
#lines(arima.total$fitted,col='#D35400',lwd=2)
#lines(sales.ts,col='#blue',lwd=2,lty=1)

legend(2018.25,9500, legend = c("Retail Sales Data", "Point Forecast for Future 24 periods",
                                "95% Confidence Interval", 
                                "80% Confidence Interval"),
       col = c("blue","#469ee0", "#dbdbdf", "#b1b5ce"), 
       lty = c(1,1,1,1),lwd =c(2,2,5,5), bty = "n")

lines(c(2021.33, 2021.33), c(1000,10000))
text(2022.5, 9550, "Future 24 Periods")
arrows(2021.5, 9250, 2023.5, 9250, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#-------------------------------------Accuracy------------------------------------------

#Linear two level forecast
round(accuracy(total.lin.pred$fitted+total.trail.ma,sales.ts),3)
#Quadratic two level forecast
round(accuracy(total.Quad.pred$fitted+total.quad.trail.ma,sales.ts),3) 
#Holt-winters optimal model
round(accuracy(hw.optimal.total$fitted,sales.ts),3)
#Two_level (Holt-Winter's Model + AR(12) fro residuals)
round(accuracy(hw.optimal.total$fitted+hw.total.residuals.ar12$fitted,sales.ts),3)
#Arima(3,2,1)(0,1,2) for entire data
round(accuracy(arima.total$fitted,sales.ts),3)
#Auto Arima For entire Data
round(accuracy(auto.total$fitted,sales.ts),3)

Accuracy_table <- rbind(round(accuracy(total.lin.pred$fitted+total.trail.ma,sales.ts),3),
                        round(accuracy(total.Quad.pred$fitted+total.quad.trail.ma,sales.ts),3) ,
                        round(accuracy(hw.optimal.total$fitted,sales.ts),3),
                        round(accuracy(hw.optimal.total$fitted+hw.total.residuals.ar12$fitted,sales.ts),3),
                        round(accuracy(arima.total$fitted,sales.ts),3),
                        round(accuracy(auto.total$fitted,sales.ts),3))

rownames(Accuracy_table) <- c("Two_level (Linear + Trailing MA for regression residuals)",
                              "Two_level (Quadratic + Trailing MA for regression residuals)",
                              "Holt-Winters Optimal Model",
                              "Two_level (Holt-Winter's Model + AR(12) fro residuals)",
                              "Arima (3,1,2)(0,1,2)",
                              "Auto Arima (2,1,1)(0,1,2)")

Accuracy_table



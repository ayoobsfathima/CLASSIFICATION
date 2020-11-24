puls = read.csv(file.choose())

#Splitting Target Class into Two Distinct Groups
puls0 = subset(puls,puls$target_class==0)
puls1 = subset(puls,puls$target_class==1)

#Z Transformation for Group 0
pulss0 = puls0[,1:8]
name = names(pulss0)
vcpuls0 = matrix(c(cov(pulss0)),nrow=8,ncol=8)
Invcpuls0 = solve(vcpuls0)
zpuls0 = Invcpuls0 %*% t(pulss0) # inverse(V-C)*t(X)
zzpuls0 = t(zpuls0)
vcZ0 = matrix(c(cov(zzpuls0)),nrow=8,ncol=8) 
colnames(zzpuls0)= name
data0=cbind(zzpuls0,target_class=0)
head(data0)

#Z Transformation for Group 1
pulss1 = puls1[,1:8]
name = names(pulss0)
vcpuls1 = matrix(c(cov(pulss1)),nrow=8,ncol=8)
Invcpuls1 = solve(vcpuls1)
zpuls1 = Invcpuls1 %*% t(pulss1)
zzpuls1 = t(zpuls1)
vcZ1 = matrix(c(cov(zzpuls1)),nrow=8,ncol=8)
colnames(zzpuls1)= name
data1=cbind(zzpuls1,target_class=1)
head(data1)

fix(vcZ0)
fix(vcZ1)

#Exporting Dataset
data = data.frame(rbind(data0,data1))
write.table(data,file="Zpulsar_star.csv",sep=",")
getwd()

#Discriminant Analysis
install.packages("MASS")
library(MASS)
fit = lda(target_class~.,data=data)
summary(fit)

plot(fit,type="both")

library(data.table)
library(ggplot2)
library(extrafont)
loadfonts(device = "all")
library(scales)
library(ggrepel)

weights <- c(100.5, 103, 104.6, 105.6,
  109.8, 110, 155.4, 212.8,
  218.7, 241.6, 377.9, 652.2)

params <- c(61100840, 20013928, 143678248, 44549160,
  126886696, 88791336, 13004888, 27161264,
  2542856, 5483032, 5288548, 66347960)

training <- c(0.733, 0.89778, 0.8477, 0.9036,
  0.96472, 0.95652, 0.7748, 0.8123,
  0.79853, 0.93115, 0.91242, 0.83959)

validation <- c(0.566, 0.76886, 0.7424, 0.7737,
  0.78842, 0.79308, 0.6977, 0.6952,
  0.67668, 0.74054, 0.77672, 0.73924)

test <- c(0.435, 0.6472, 0.619, 0.6557,
  0.6654, 0.675, 0.5786, 0.5755,
  0.5471, 0.6048, 0.6573, 0.6177)

training <- 1-training
validation <- 1-validation
test <- 1-test

nn.name <- c('AlexNet',
  'DenseNet-201',
  'VGG-19-bn',
  'ResNet-101',
  'Wide ResNet-101-2',
  'ResNeXt-101-32x8d',
  'GoogLeNet','Inception v3',
  'MobileNet v3 small',
  'MobileNet v3 large',
  'EfficientNet-b0',
  'EfficientNet-b7')

overfitting <- data.table(params=weights,`Training Error`=training,`Validation Error  `=validation,`New Test Error`=test,nn.name=nn.name)

colors <- c("Training Error" = "dodgerblue","Validation Error  " = "springgreen3", "New Test Error" = "firebrick2")

png(filename="overfit_norm.png",height=2000,width=3000)
gg <- ggplot(overfitting,aes(x=params)) +
  geom_line(aes(y=`Training Error`,color="Training Error"),size=3) + 
  geom_line(aes(y=`Validation Error  `,color="Validation Error  "),size=3) + 
  geom_line(aes(y=`New Test Error`,color="New Test Error"),size=3) + 
  geom_text_repel(color="black",hjust=-0,vjust=-0,size=15,y=training,label=nn.name,fontface="bold",family="Times New Roman") +
  xlab("Model Complexity (norm of weights)") + ylab("Prediction Error") +
  ylim(0,0.6) + theme_bw() +
   scale_x_continuous(limits=c(0,700)) +
   theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),
    text = element_text(size = 100, family="Times New Roman"), plot.margin=unit(c(1,3,1,1), "cm"),
    legend.title=element_blank(),axis.text.x=element_text(margin=margin(t=20,r=0,b=0,l=0)),
    axis.text.y=element_text(margin=margin(t=0,r=20,b=0,l=0)),
    axis.ticks = element_blank(),legend.position = c(0.857,0.913),legend.key.height=unit(3.5,"cm"),legend.key.width = unit(5,"cm"),legend.key.size=unit(4,"cm"),
    axis.title.y=element_text(margin=margin(t=0,r=30,b=0,l=0)),legend.margin=margin(t=10,r=10,b=30,l=30),
    axis.title.x=element_text(margin=margin(t=40,r=0,b=0,l=0)),
    legend.box.background = element_rect(colour = "black",size=2)) + scale_color_manual(values=colors)
print(gg)
dev.off()


png(filename="overfit_norm.png",height=2000,width=1600)
gg <- ggplot(overfitting,aes(x=params, y=training, color="blue",label=nn.name)) +
  geom_line(color="blue",size=3) + 
  geom_line(y=validation,color="green",size=3) + 
  geom_line(y=test,color="red",size=3) + geom_text(hjust=-0,vjust=-0,size=15,color="blue") +
  xlab("Model Complexity (norm of weights)") + ylab("Prediction Error") +
  ylim(0,0.6) + theme_bw() + 
   scale_x_continuous(labels = label_number(limits=c(100, 700)) +
   theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),legend.position="topright",
    text = element_text(size = 90, family="Times New Roman"), plot.margin=unit(c(1,3,1,1), "cm"),
    axis.text.x=element_text(margin=margin(t=20,r=0,b=0,l=0)),
    axis.text.y=element_text(margin=margin(t=0,r=20,b=0,l=0)),
    axis.ticks = element_blank(),
    axis.title.y=element_text(margin=margin(t=0,r=30,b=0,l=0)),
    axis.title.x=element_text(margin=margin(t=30,r=0,b=0,l=0)))
print(gg)
dev.off()

png(file="overfit_norm.png",width=2000,height=1600)
par(mar=c(14,16,1,1),mgp=c(12,5,0))
ylims=c(0,0.7)
matplot(params,training,type="n",col=colors()[558],xlab="Model Complexity (norm of weights)",ylab="Prediction Error",ylim=ylims,lty=1,
  cex.lab=5,cex.axis=5)
lines(params,training,col=2,lwd=4)
lines(params,validation,col=3,lwd=4)
lines(params,test,col=4,lwd=4)
legend(x = "topright",
        cex=5,          # Position
       ncol = 1,
       legend = c("Training Error","Validation Error","New Test Error"),
       lty = c(1, 1, 1),           # Line types
       col = c(4,3,2),           # Line colors
       lwd = c(4,4,4))
dev.off()
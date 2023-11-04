import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[2,5,7,9,7]
plt.plot(x,y)
plt.xticks([1,3,5,15],fontsize=12,rotation=45)
plt.show()
plt.waitforbuttonpress(0)
print("Aksak")
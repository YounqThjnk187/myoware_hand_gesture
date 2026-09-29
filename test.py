import control as ct
import matplotlib.pyplot as plt 
#G(s) = 1 / (s^3 + 3s^2 + 2s)
sys = ct.tf([1], [1, 3, 2, 0])  # Hệ thống điều khiển bậc hai
ct.root_locus(sys)
plt.show()
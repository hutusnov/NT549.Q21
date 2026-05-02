import pandas as pd
import matplotlib.pyplot as plt

# Đọc file kết quả lúc mạng bị lag
df = pd.read_csv("benchmark_chaos.csv")

# Lọc lấy 2 thuật toán cần so sánh
df_filtered = df[df['strategy'].isin(['RoundRobin', 'DQN_Agent'])]
strategies = df_filtered['strategy'].tolist()
sla_rates = df_filtered['sla_violation_rate'].astype(float).tolist()

# Đổi tên cho đẹp
display_names = ["RoundRobin\n(Mù quáng chia đều)", "Dueling DQN\n(Nhận diện lag & Bẻ lái)"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(display_names, sla_rates, color=['#E24B4A', '#1D9E75'], width=0.45)

for bar, v in zip(bars, sla_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 2, 
             f"{v:.1f}%", ha='center', fontweight='bold', fontsize=12)

ax.set_ylabel("Tỷ lệ vi phạm SLA (%)", fontweight='bold')
ax.set_ylim(0, max(sla_rates) + 20)
ax.set_title("Kịch bản 3: Tỷ lệ Vi phạm SLA khi sự cố Mạng (Chaos Engineering)", fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig("p2_chaos_sla_comparison.png", dpi=300)
print("✅ Đã vẽ biểu đồ SLA: p2_chaos_sla_comparison.png")

#!/bin/bash
cd "$(dirname "$0")"

echo "🧠 数学教材智能学习系统"
echo "========================"

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "❌ 未找到虚拟环境，请先运行: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "🚀 启动中..."
sleep 1

# Launch app in background, capture URL
python app.py &
APP_PID=$!

# Wait for server to be ready and open browser
echo "⏳ 等待服务启动..."
for i in $(seq 1 30); do
    if curl -s http://localhost:7860 > /dev/null 2>&1; then
        echo "✅ 服务已就绪"
        open http://localhost:7860
        break
    fi
    sleep 1
done

# Wait for app process
wait $APP_PID

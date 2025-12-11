#!/bin/bash
 
# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!
 
# Pause for Ollama to start.
sleep 5
 
echo "🔴 Retrieve deepseek-ocr:3b model..."
ollama pull deepseek-ocr:3b
echo "🟢 Done!"

echo "🔴 Retrieve qwen3:8b model..."
ollama pull qwen3:8b
echo "🟢 Done!"

echo "🔴 Retrieve qwen3-vl:8b model..."
ollama pull qwen3-vl:8b
echo "🟢 Done!"

 
# Wait for Ollama process to finish.
wait $pid


#!/bin/bash
 
# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!
 
# Pause for Ollama to start.
sleep 5
 
echo "Retrieve qwen3.5:35b model (Text LLM)..."
ollama pull qwen3.5:35b
echo "Done!"

echo "Retrieve qwen3-vl:32b model (Vision LLM)..."
ollama pull qwen3-vl:32b
echo "Done!"

 
# Wait for Ollama process to finish.
wait $pid

# Installation instructions for pyaudio:
# APPLE Mac OS X
#   brew install portaudio
#   pip install pyaudio
# Debian/Ubuntu
#   sudo apt-get install python-pyaudio python3-pyaudio
#   or
#   pip install pyaudio
# CentOS
#   sudo yum install -y portaudio portaudio-devel && pip install pyaudio
# Microsoft Windows
#   python -m pip install pyaudio

import os
import json
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
import datetime
import pytz
import subprocess
import platform

# Load environment variables from .env file
load_dotenv()

# Function to check current time
def check_current_time(location=None):
    # Use pytz to get NYC time
    nyc_timezone = pytz.timezone('America/New_York')
    current_time = datetime.datetime.now(nyc_timezone)
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    if location:
        print(f"Current time in {location} (NYC timezone): {formatted_time}")
        return {
            "current_time": formatted_time,
            "location": location,
            "timezone": "America/New_York"
        }
    else:
        print(f"Current time in NYC: {formatted_time}")
        return {
            "current_time": formatted_time,
            "timezone": "America/New_York"
        }

# Sample function to get weather information
def get_current_temperature(location):
    # This is a mock function - in a real application, you would call a weather API
    return {
        "temperature": 26.1,
        "location": location,
        "unit": "celsius"
    }

# Function to get a function by name
def get_function_by_name(name):
    function_map = {
        "check_current_time": check_current_time,
        "get_current_temperature": get_current_temperature
    }
    return function_map.get(name)

# Function to play audio file
def play_audio(file_path):
    print(f"Playing audio file: {file_path}")
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", file_path])
        elif system == "Linux":
            subprocess.run(["aplay", file_path])
        elif system == "Windows":
            from winsound import PlaySound, SND_FILENAME
            PlaySound(file_path, SND_FILENAME)
        else:
            print(f"Unsupported operating system: {system}")
    except Exception as e:
        print(f"Error playing audio: {e}")

# Define tools for Qwen Omni
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_current_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for, e.g., 'San Francisco, CA, USA'"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Example 1: Basic text and audio generation
def example_text_audio():
    print("\n=== Example 1: Basic Text and Audio Generation ===")
    completion = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=[{"role": "user", "content": "你是谁"}],
        # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
        modalities=["text", "audio"],
        audio={"voice": "Cherry", "format": "wav"},
        # stream 必须设置为 True，否则会报错
        stream=True,
        stream_options={"include_usage": True},
    )

    # 方式1: 待生成结束后再进行解码
    audio_string = ""
    for chunk in completion:
        if chunk.choices:
            if hasattr(chunk.choices[0].delta, "audio"):
                try:
                    audio_string += chunk.choices[0].delta.audio["data"]
                except Exception as e:
                    print(chunk.choices[0].delta.audio["transcript"])
        else:
            print(chunk.usage)

    wav_bytes = base64.b64decode(audio_string)
    audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
    audio_file = "audio_assistant_py.wav"
    sf.write(audio_file, audio_np, samplerate=24000)
    print(f"Audio saved to {audio_file}")
    
    # Automatically play the audio file
    play_audio(audio_file)

# Example 2: Function calling with Qwen Omni
def example_function_calling():
    print("\n=== Example 2: Function Calling with Qwen Omni ===")
    
    # Initial messages
    messages = [
        {
            "role": "system",
            "content": f"You are Qwen Omni, a helpful assistant. Current Date: {datetime.datetime.now().strftime('%Y-%m-%d')}"
        },
        {
            "role": "user",
            "content": "What time is it now in New York?"
        }
    ]
    
    # First call to get tool calls
    print("Sending initial request to model...")
    response = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=messages,
        tools=TOOLS,
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        stream=True,  # Required for Qwen Omni
        stream_options={"include_usage": True}
    )
    
    # Process streaming response
    print("\nProcessing streaming response...")
    assistant_message = {"role": "assistant", "content": "", "tool_calls": []}
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            # Accumulate content
            if hasattr(delta, "content") and delta.content is not None:
                assistant_message["content"] += delta.content
            
            # Accumulate tool calls
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    # Find if we already have this tool call
                    existing_tool_call = next((tc for tc in assistant_message["tool_calls"] 
                                             if tc.get("id") == tool_call.id), None)
                    
                    if existing_tool_call is None:
                        # New tool call
                        assistant_message["tool_calls"].append({
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments or ""
                            }
                        })
                    else:
                        # Update existing tool call
                        index = assistant_message["tool_calls"].index(existing_tool_call)
                        if tool_call.function.arguments:
                            assistant_message["tool_calls"][index]["function"]["arguments"] += tool_call.function.arguments
        else:
            # This is usage information
            if hasattr(chunk, "usage"):
                print(f"Usage: {chunk.usage}")
    
    # Add the model's response to messages
    messages.append(assistant_message)
    
    print("\nModel response with tool calls:")
    print(assistant_message)
    
    # Process tool calls and add results to messages
    if tool_calls := assistant_message.get("tool_calls", []):
        print("\nProcessing tool calls...")
        for tool_call in tool_calls:
            call_id = tool_call["id"]
            if fn_call := tool_call.get("function"):
                fn_name = fn_call["name"]
                try:
                    # Fix malformed JSON by removing any leading {}
                    args_str = fn_call["arguments"]
                    if args_str.startswith("{}"):
                        args_str = args_str[2:]
                    
                    # Parse the arguments
                    fn_args = json.loads(args_str)
                    
                    # Get the function to call
                    func = get_function_by_name(fn_name)
                    if not func:
                        print(f"Function {fn_name} not found")
                        continue
                        
                    print(f"Calling function: {fn_name} with args: {fn_args}")
                    
                    # Call the function with the arguments
                    if fn_name == "check_current_time" and "location" in fn_args:
                        # Special handling for check_current_time with location
                        fn_res = json.dumps(check_current_time(location=fn_args.get("location")))
                    else:
                        # Normal function call
                        fn_res = json.dumps(func(**fn_args))
                    
                    messages.append({
                        "role": "tool",
                        "content": fn_res,
                        "tool_call_id": call_id,
                    })
                except json.JSONDecodeError as e:
                    print(f"Error parsing arguments: {e}")
                    print(f"Raw arguments: {fn_call['arguments']}")
                    
                    # Try to extract a valid JSON object from the string
                    import re
                    json_pattern = r'\{[^\{\}]*\}'
                    matches = re.findall(json_pattern, fn_call["arguments"])
                    if matches:
                        try:
                            fn_args = json.loads(matches[0])
                            print(f"Extracted arguments: {fn_args}")
                            print(f"Calling function: {fn_name} with args: {fn_args}")
                            fn_res = json.dumps(get_function_by_name(fn_name)(**fn_args))
                            
                            messages.append({
                                "role": "tool",
                                "content": fn_res,
                                "tool_call_id": call_id,
                            })
                        except Exception as e2:
                            print(f"Error processing extracted arguments: {e2}")
    
    # Second call to get final response with audio
    print("\nSending follow-up request with tool results...")
    response = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=messages,
        tools=TOOLS,
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        modalities=["text", "audio"],  # Request both text and audio
        audio={"voice": "Cherry", "format": "wav"},
        stream=True,  # Required for Qwen Omni
        stream_options={"include_usage": True}
    )
    
    # Process final streaming response with audio
    final_response = ""
    audio_string = ""
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            # Accumulate text content
            if hasattr(delta, "content") and delta.content is not None:
                final_response += delta.content
                print(delta.content, end="", flush=True)
            
            # Accumulate audio content
            if hasattr(delta, "audio"):
                try:
                    audio_string += delta.audio["data"]
                except Exception as e:
                    print("\nAudio transcript:", delta.audio["transcript"])
        else:
            # This is usage information
            if hasattr(chunk, "usage"):
                print(f"\nUsage: {chunk.usage}")
    
    print("\n\nFinal response:")
    print(final_response)
    
    # Save audio if we received any
    if audio_string:
        print("\nSaving audio response...")
        wav_bytes = base64.b64decode(audio_string)
        audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
        audio_file = "function_call_response.wav"
        sf.write(audio_file, audio_np, samplerate=24000)
        print(f"Audio saved to {audio_file}")
        
        # Automatically play the audio file
        play_audio(audio_file)

# Main execution
if __name__ == "__main__":
    try:
        # Run the examples
        # example_text_audio()
        example_function_calling()
        
        # Call the function to check current time directly
        print("\n=== Direct Function Call ===")
        print(check_current_time())
    except Exception as e:
        print(f"\nError: {e}")
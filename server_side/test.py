import sys
import inspect
from faster_whisper import WhisperModel

# Print the version of faster_whisper
print(f"faster_whisper version: {getattr(sys.modules['faster_whisper'], '__version__', 'unknown')}")

# Create a model instance
model = WhisperModel("base", device="cuda", compute_type="float32")

# Explore the transcribe method
print("\n=== WhisperModel.transcribe method ===")
transcribe_signature = inspect.signature(model.transcribe)
print(f"Signature: {transcribe_signature}")
print("\nParameters:")
for param_name, param in transcribe_signature.parameters.items():
    print(f"  {param_name}: {param.annotation}")
    if param.default is not inspect.Parameter.empty:
        print(f"    Default: {param.default}")

# Check if there's a callback parameter
has_callback = any(param for param in transcribe_signature.parameters.values() 
                  if "callback" in param.name.lower())
print(f"\nHas callback parameter: {has_callback}")

# Check for any other progress-related parameters
progress_params = [param for param in transcribe_signature.parameters.keys() 
                  if any(kw in param.lower() for kw in ["progress", "callback", "hook", "on_"])]
if progress_params:
    print(f"Potential progress-related parameters: {progress_params}")

# Try to find documentation
print("\n=== Documentation ===")
print(model.transcribe.__doc__ or "No docstring available")

# Check what the transcribe method returns
print("\n=== Return value exploration ===")
print("Transcribing a short sample to examine return values...")
try:
    # Create a short silent audio file for testing
    import numpy as np
    import soundfile as sf
    
    # Generate 1 second of silence
    sample_rate = 16000
    silence = np.zeros(sample_rate)
    sf.write('silence.wav', silence, sample_rate)
    
    # Transcribe it
    result = model.transcribe('silence.wav')
    
    # Examine the result
    print(f"Return type: {type(result)}")
    print(f"Return value contains: {dir(result[0])[:10]}... (truncated)")
    
    # Check if the segments object has any progress-related methods
    segments = result[0]
    segment_methods = [method for method in dir(segments) 
                      if callable(getattr(segments, method)) and not method.startswith('_')]
    print(f"Segment methods: {segment_methods}")
    
except Exception as e:
    print(f"Error during test transcription: {e}")
finally:
    import os
    if os.path.exists('silence.wav'):
        os.remove('silence.wav')

print("\nInvestigation complete!")
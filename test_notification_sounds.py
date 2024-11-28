import numpy as np
import sounddevice as sd
import time
import argparse

# Different sound sets to try
SOUND_SETS = {
    "musical": {
        "start": (523.25, 0.15),  # C5 - clear, bright
        "stop": (659.25, 0.15),  # E5 - higher, distinct
        "complete": (783.99, 0.15),  # G5 - success feeling
        "error": (311.13, 0.15),  # Eb4 - low, warning
        "empty": (392.00, 0.15),  # G4 - medium, notice
    },
    "ascending": {
        "start": (392.00, 0.15),  # G4
        "stop": (493.88, 0.15),  # B4
        "complete": (587.33, 0.15),  # D5
        "error": (329.63, 0.15),  # E4
        "empty": (440.00, 0.15),  # A4
    },
    "gentle": {
        "start": (415.30, 0.15),  # Ab4 - soft
        "stop": (466.16, 0.15),  # Bb4 - gentle
        "complete": (554.37, 0.15),  # Db5 - pleasant
        "error": (277.18, 0.15),  # Db4 - low
        "empty": (369.99, 0.15),  # F#4 - neutral
    },
}


def generate_beep(frequency, duration):
    """Generate a simple beep sound with smooth envelope"""
    rate = 44100
    t = np.linspace(0, duration, int(rate * duration), False)

    # Create the basic sine wave
    samples = np.sin(2 * np.pi * frequency * t)

    # Create smooth envelope
    attack_time = 0.02  # 20ms attack
    release_time = 0.15  # 50ms release

    attack_len = int(attack_time * rate)
    release_len = int(release_time * rate)

    # Apply attack (fade in)
    samples[:attack_len] *= np.linspace(0, 1, attack_len)

    # Apply release (fade out)
    samples[-release_len:] *= (
        np.linspace(1, 0, release_len) ** 2
    )  # Quadratic fade for smoother release

    # Normalize to prevent clipping
    samples = samples * 0.7  # Reduce volume slightly

    return samples.astype(np.float32)


def play_sound(frequency, duration, repeats=1):
    """Play a notification beep"""
    try:
        samples = generate_beep(frequency, duration)
        for _ in range(repeats):
            sd.play(samples, 44100)
            sd.wait()
            if repeats > 1:
                time.sleep(0.15)  # Longer gap between repeats
    except Exception as e:
        print(f"Error playing sound: {e}")


def test_sound_set(name, sounds):
    """Test a complete set of notification sounds"""
    print(f"\nTesting sound set: {name}")
    for sound_type, (freq, dur) in sounds.items():
        print(f"Playing {sound_type} sound...")
        repeats = 2 if sound_type in ["error", "empty"] else 1
        play_sound(freq, dur, repeats)
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Test notification sound sets")
    parser.add_argument(
        "--set", choices=list(SOUND_SETS.keys()), help="Specific sound set to test"
    )
    args = parser.parse_args()

    if args.set:
        test_sound_set(args.set, SOUND_SETS[args.set])
    else:
        for name, sounds in SOUND_SETS.items():
            test_sound_set(name, sounds)


if __name__ == "__main__":
    main()

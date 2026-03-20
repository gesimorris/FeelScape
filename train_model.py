"""
This is where the data is trained. The training pipeline is ran with the original pairs of image and MIDI files,
and then augmented to create a larger dataset for better accuracy.
The model is trained with the specified hyperparameters, and the trained model is saved to the output directory.
If there are any issues during training, they will be caught and printed to the console for debugging.

"""

import sys
sys.path.append('backend')

from training_pipeline import run_complete_training_pipeline

training_pairs = [
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000003_(2).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 7 - E Min.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000002_(3).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 1 - C Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000004_(5).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 2 - C Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000005_(7).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 4 - D Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000031_(3).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/1.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000004_(7).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 1 - C Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000005_(3).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 2 - C Min.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000010_(3).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 3 - D Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000010_(4).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 4 - D Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000032_(3).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 6 - D Min.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000032.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 7 - E Min.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000033_(5).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 8 - F Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000034_(5).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 14 - F Min.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000005_(5).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 11 - A Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000894_(2).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 22 - B Min.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000889_(3).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 15 - A Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000887.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 15 - A Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000885.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 1 - C Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000884.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 6 - D Min.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000884_(2).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 2 - C Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000883_(2).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 4 - D Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000881_(2).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Lofi Piano MIDI.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000878_(2).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 18 - G Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000877.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Lofi Piano MIDI.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000875_(2).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 15 - A Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000874.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 18 - G Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000865_(3).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 12 - E Min.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000854.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 11 - E Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000848.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 11 - E Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000847.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 11 - E Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000844.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 11 - E Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000841.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 11 - E Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000843.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Lofi MIDI 11 - E Maj.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000839.jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 7 - E Min.mid'},
    {'image_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/ImagesAI/00000834_(2).jpg', 'midi_path': '/Users/gesimorris-odubo/Downloads/lofi-generator/MIDI/Cymatics - Eternity MIDI 7 - E Min.mid'},
]

if len(training_pairs) < 10:
    print("WARNING: You have fewer than 10 training pairs.")
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        sys.exit(0)

print(f"\n{'='*70}")
print(f"LOFI GENERATOR - MODEL TRAINING")
print(f"{'='*70}")
print(f"\nTraining with {len(training_pairs)} original pairs")
print(f"Will augment to 1000+ pairs for better accuracy\n")

try:
    model, scaler_x, scaler_y, history = run_complete_training_pipeline(
        original_pairs=training_pairs,
        
        augment=True,                    
        augmentation_target=1000,        
        
        test_size=0.15,                  
        val_size=0.15,                   
        
        
        hidden_sizes=[64, 128, 128, 64], # 4 hidden layers
        
        
        learning_rate=0.001,             
        dropout_rate=0.3,               
        epochs=2000,                     
        batch_size=32,                   
        early_stopping_patience=100,
        
        output_dir='./backend/models'
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nYour model is ready to use!")
    print("\nHappy music generating!")
    
except Exception as e:
    print("\n" + "="*70)
    print("TRAINING FAILED")
    print("="*70)
    print(f"\nError: {e}")
    print("\nPlease check:")
    print("1. All image and MIDI file paths are correct")
    print("2. All files exist and are readable")
    print("3. You have enough disk space")
    print("4. All dependencies are installed")
    import traceback
    traceback.print_exc()
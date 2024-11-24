import argparse
import numpy as np
import os
import json
import librosa
import soundfile as sf
from feature_extractor import FeatureExtractor
from train import CryDetectionModel
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self, model_path: str):
        """Initialize the analyzer with a trained model."""
        self.model = CryDetectionModel.load(model_path)
        self.config = self._load_config(model_path)
        
    def _load_config(self, model_path: str) -> Dict:
        """Load experiment configuration if available."""
        config_path = os.path.join(os.path.dirname(model_path), 'experiment_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
        
    def analyze_audio(self, audio_path: str, output_dir: Optional[str] = None) -> Tuple[int, float, Dict]:
        """
        Analyze an audio file and optionally save visualization.
        Returns: (prediction, probability, analysis_results)
        """
        try:
            # Load and preprocess audio
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Make prediction
            prediction, probability = self.model.predict(audio_path)
            
            # Generate analysis results
            analysis_results = {
                'duration': duration,
                'sample_rate': sr,
                'prediction': 'Cry' if prediction == 1 else 'Not cry',
                'probability': probability,
                'audio_path': audio_path
            }
            
            # Generate and save visualization if output_dir is provided
            if output_dir:
                self._save_visualization(y, sr, analysis_results, output_dir)
                
            return prediction, probability, analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing audio file {audio_path}: {str(e)}")
            raise
            
    def _save_visualization(self, y: np.ndarray, sr: int, results: Dict, output_dir: str):
        """Generate and save audio visualization."""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        
        # Mel spectrogram
        plt.subplot(3, 1, 2)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        # Add prediction results
        plt.subplot(3, 1, 3)
        plt.axis('off')
        info_text = f"""
        Analysis Results:
        Prediction: {results['prediction']}
        Confidence: {results['probability']:.2f}
        Duration: {results['duration']:.2f} seconds
        Sample Rate: {results['sample_rate']} Hz
        Model Type: {self.config.get('model_type', 'Unknown')}
        """
        plt.text(0.1, 0.5, info_text, fontsize=12, va='center')
        
        # Save plot
        base_name = os.path.splitext(os.path.basename(results['audio_path']))[0]
        plot_path = os.path.join(output_dir, f'{base_name}_analysis.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_path}")

def process_batch(analyzer: AudioAnalyzer, input_path: str, output_dir: str) -> Dict:
    """Process multiple audio files and generate summary."""
    results = []
    
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    
    for audio_file in files:
        try:
            prediction, probability, analysis = analyzer.analyze_audio(audio_file, output_dir)
            results.append({
                'file': audio_file,
                'prediction': 'Cry' if prediction == 1 else 'Not cry',
                'probability': probability
            })
            logger.info(f"Processed {audio_file}: {results[-1]['prediction']} "
                       f"(probability: {probability:.2f})")
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {str(e)}")
            
    # Generate summary
    summary = {
        'total_files': len(files),
        'processed_files': len(results),
        'cry_detected': sum(1 for r in results if r['prediction'] == 'Cry'),
        'not_cry_detected': sum(1 for r in results if r['prediction'] == 'Not cry'),
        'results': results
    }
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    return summary

def main():
    parser = argparse.ArgumentParser(description='Cry detection inference')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input audio file or directory')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                      help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = AudioAnalyzer(args.model_path)
        
        # Process audio files and generate summary
        summary = process_batch(analyzer, args.input_path, args.output_dir)
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total files: {summary['total_files']}")
        print(f"Successfully processed: {summary['processed_files']}")
        print(f"Cry detected: {summary['cry_detected']}")
        print(f"Not cry detected: {summary['not_cry_detected']}")
        print(f"\nDetailed results saved to: {args.output_dir}/analysis_summary.json")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
#ott_integration.py

import os
import json
import time
from datetime import datetime

class OTTIntegration:
    def __init__(self):
        print(" INITIALIZING OTT PLATFORM INTEGRATION")
        
        # OTT platform configuration
        self.ott_config = {
            'api_endpoint': os.getenv('OTT_API_ENDPOINT', 'https://api.ott-platform.com/v1'),
            'stream_key': os.getenv('OTT_STREAM_KEY', 'demo_stream'),
            'max_retries': 3,
            'timeout': 5.0
        }
        
        # Integration status
        self.is_connected = False
        self.connection_stats = {
            'total_sent': 0,
            'failed_attempts': 0,
            'last_success': None
        }
        
        # Stream management
        self.active_streams = {}
        
        print(" OTT Integration initialized")
    
    def connect_to_platform(self, stream_config=None):
        """Connect to OTT platform"""
        print("\n CONNECTING TO OTT PLATFORM")
        print("=" * 40)
        
        try:
            # Simulate connection to OTT platform
            print(" Establishing connection to OTT API...")
            time.sleep(1)  # Simulate connection time
            
           
            if stream_config:
                self.ott_config.update(stream_config)
            
            
            test_result = self._test_connection()
            
            if test_result:
                self.is_connected = True
                self.connection_stats['last_success'] = datetime.now()
                print(" Successfully connected to OTT platform")
                print(f" Stream Key: {self.ott_config['stream_key']}")
                return True
            else:
                print(" Failed to connect to OTT platform")
                return False
                
        except Exception as e:
            print(f" Connection error: {e}")
            return False
    
    def _test_connection(self):
        """Test connection to OTT platform"""
        try:
            # Simulate API test
            print(" Testing OTT API connection...")
            time.sleep(0.5)
            
            # In real implementation, this would be an actual API call
            return True  # Simulate success
            
        except Exception as e:
            print(f" Connection test failed: {e}")
            return False
    
    def send_translation_to_feed(self, translation_data, stream_id="main"):
        """Send translated content to OTT feed"""
        if not self.is_connected:
            print("  Not connected to OTT platform")
            return False
        
        try:
            # Prepare OTT payload
            ott_payload = self._prepare_ott_payload(translation_data, stream_id)
            
            # Send to OTT platform
            success = self._send_to_ott_api(ott_payload)
            
            if success:
                self.connection_stats['total_sent'] += 1
                self.connection_stats['last_success'] = datetime.now()
                print(f" OTT Feed: Sent translation #{self.connection_stats['total_sent']}")
                return True
            else:
                self.connection_stats['failed_attempts'] += 1
                print(f" Failed to send to OTT feed (attempt {self.connection_stats['failed_attempts']})")
                return False
                
        except Exception as e:
            print(f" OTT send error: {e}")
            self.connection_stats['failed_attempts'] += 1
            return False
    
    def _prepare_ott_payload(self, translation_data, stream_id):
        """Prepare payload for OTT platform"""
        original = translation_data['original']
        translations = translation_data['translations']
        
        payload = {
            'metadata': {
                'stream_id': stream_id,
                'sequence_id': self.connection_stats['total_sent'] + 1,
                'timestamp': translation_data['timestamp'],
                'source_language': original['language'],
                'target_languages': list(translations.keys()),
                'latency': translation_data['latency']
            },
            'content': {
                'original_text': original['text'],
                'translations': translations
            },
            'delivery': {
                'format': 'text',
                'priority': 'high',
                'ttl': 300
            }
        }
        
        return payload
    
    def _send_to_ott_api(self, payload):
        """Send payload to OTT API"""
        try:
            # Simulate API call
            print(f" Sending to OTT API... (Stream: {payload['metadata']['stream_id']})")
            time.sleep(0.1)  # Simulate network delay
            
            # Simulate successful send
            return True
            
        except Exception as e:
            print(f" API send error: {e}")
            return False
    
    def create_stream(self, stream_name, config=None):
        """Create a new stream on OTT platform"""
        print(f"\n CREATING OTT STREAM: {stream_name}")
        
        try:
            stream_config = {
                'name': stream_name,
                'type': 'translation_feed',
                'languages': ['en', 'hi'] + self._get_target_languages(),
                'created_at': datetime.now().isoformat()
            }
            
            if config:
                stream_config.update(config)
            
            # Register stream with OTT platform
            self.active_streams[stream_name] = stream_config
            
            print(f" Stream created: {stream_name}")
            print(f" Configuration: {json.dumps(stream_config, indent=2)}")
            
            return stream_name
            
        except Exception as e:
            print(f" Stream creation error: {e}")
            return None
    
    def _get_target_languages(self):
        """Get target languages from translator"""
        try:
            from translator import EnhancedTranslator
            translator = EnhancedTranslator()
            # Return all except source languages
            return [lang for lang in translator.supported_languages.keys() if lang not in ['en', 'hi']]
        except:
            return ['es', 'fr', 'de', 'ja', 'zh']  # Fallback
    
    def get_connection_status(self):
        """Get current connection status and statistics"""
        status = {
            'connected': self.is_connected,
            'statistics': self.connection_stats.copy(),
            'active_streams': list(self.active_streams.keys()),
            'config': {
                'api_endpoint': self.ott_config['api_endpoint'],
                'stream_key': self.ott_config['stream_key']
            }
        }
        
        if self.connection_stats['last_success']:
            status['last_success'] = self.connection_stats['last_success'].isoformat()
        
        return status
    
    def disconnect(self):
        """Disconnect from OTT platform"""
        print("\n🔌 DISCONNECTING FROM OTT PLATFORM")
        
        self.is_connected = False
        self.active_streams.clear()
        
        print(" Disconnected from OTT platform")
        print(f" Final stats: {self.connection_stats['total_sent']} sent, "
              f"{self.connection_stats['failed_attempts']} failed")

# Backward compatibility
OTTIntegrationManager = OTTIntegration

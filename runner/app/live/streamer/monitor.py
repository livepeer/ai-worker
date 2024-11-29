import os
import json
import time
import logging
from confluent_kafka import Producer


class KafkaProducer:
    def __init__(self):
        config = {
            'bootstrap.servers': os.getenv("BOOTSTRAP_SERVERS"),
            'sasl.mechanisms': 'PLAIN',
            'security.protocol': 'SASL_SSL',
            'sasl.username': os.getenv("KAFKA_USERNAME"),
            'sasl.password': os.getenv("KAFKA_PASSWORD")
        }
        self.producer = Producer(config)
        
    def _delivery_report(self, err, msg):
        if err is not None:
            logging.error(f'Message delivery failed: {err}')
        else:
            logging.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')
    
    def send_event(self, topic, event_type, data):
        event = {
            'type': event_type,
            'timestamp': str(int(time.time() * 1000)),
            'data': data
        }
        
        try:
            payload = json.dumps(event).encode('utf-8')
            
            self.producer.produce(
                topic, 
                value=payload,
                callback=self._delivery_report
            )
            
            self.producer.poll(0)
        
        except Exception as e:
            logging.error(f'Failed to send event: {e}')
    
    def flush(self, timeout=10):
        self.producer.flush(timeout)
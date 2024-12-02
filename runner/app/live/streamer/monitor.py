import os
import json
import time
import logging
import asyncio
from confluent_kafka import Producer


class KafkaProducer:
    def __init__(self):
        config = {
            "bootstrap.servers": os.getenv("BOOTSTRAP_SERVERS"), # figure out a better way to fetch them as secrets?
            "sasl.mechanisms": "PLAIN",
            "security.protocol": "SASL_SSL",
            "sasl.username": os.getenv("KAFKA_USERNAME"),
            "sasl.password": os.getenv("KAFKA_PASSWORD")
        }
        try:
            self.producer = Producer(config)
            self._running = False
        except Exception as e:
            self.producer = None
            logging.error(f"Error starting a producer with: {e}")
    
    def start(self):
        if self.producer is None:
            logging.error("No producer instance available")
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_producer())
    
    async def stop(self):
        self._running = False
        if self.producer is None:
            logging.error("No producer instance available or stopped")
            return
        await self.flush()
        if hasattr(self, "_poll_task"):
            await self._poll_task
    
    async def _poll_producer(self):
        while self._running:
            self.producer.poll(0)
            await asyncio.sleep(2)
    
    def _delivery_report(self, err, msg):
        if err is not None:
            logging.error(f"Message delivery failed: {err}")
        else:
            logging.error(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def send_event(self, topic, event_type, data):
        if self.producer is None or not self._running:
            logging.error("No producer instance available or stopped")
            return
        
        event = {
            "type": event_type,
            "timestamp": str(int(time.time() * 1000)),
            "data": data
        }
        
        try:
            payload = json.dumps(event).encode("utf-8")
            self.producer.produce(
                topic, 
                value=payload,
                callback=self._delivery_report
            )
            
        except Exception as e:
            logging.error(f"Failed to send event: {e}")
    
    async def flush(self, timeout=10):
        if self.producer is None or not self._running:
            logging.error("No producer instance available or stopped")
            return
        await asyncio.to_thread(self.producer.flush, timeout)
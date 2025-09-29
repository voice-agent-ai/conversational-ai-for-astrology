
import asyncio, os
from videosdk.agents import Agent, AgentSession, CascadingPipeline, JobContext, RoomOptions, WorkerJob,ConversationFlow
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from typing import AsyncIterator

# Pre-downloading the Turn Detector model
pre_download_model()

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""You are a wise and engaging astrology guide with a deep understanding of astrological concepts and practices. Your primary role is to provide insightful astrological interpretations, answer inquiries about zodiac signs, horoscopes, planetary movements, and offer guidance on how these elements might influence personal and professional life. Maintain an empathetic, encouraging, and mystical tone, ensuring users feel supported and enlightened. Engage users with clear explanations and personalized insights, tailored to their specific astrological profiles. You should be proficient in discussing astrology's historical context, modern applications, and how users can harness astrological knowledge for self-improvement. Avoid offering definitive predictions or life-altering advice; instead, encourage exploration and self-reflection.""")
    async def on_enter(self): await self.session.say("Hello! How can I help you today regarding conversational ai for astrology?")
    async def on_exit(self): await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    # Create agent and conversation flow
    agent = MyVoiceAgent()
    conversation_flow = ConversationFlow(agent)

    # Create pipeline
    pipeline = CascadingPipeline(
        stt=DeepgramSTT(model="nova-2", language="en"),
        llm=OpenAILLM(model="gpt-4o"),
        tts=ElevenLabsTTS(model="eleven_flash_v2_5"),
        vad=SileroVAD(threshold=0.35),
        turn_detector=TurnDetector(threshold=0.8)
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    try:
        await context.connect()
        await session.start()
        # Keep the session running until manually terminated
        await asyncio.Event().wait()
    finally:
        # Clean up resources when done
        await session.close()
        await context.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions(
     #  room_id="YOUR_MEETING_ID",  # Set to join a pre-created room; omit to auto-create
        name="VideoSDK Cascaded Agent for conversational ai for astrology",
        playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()

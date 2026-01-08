from zhipuai import ZhipuAI

from env_utils import ZHIPU_API_KEY

client = ZhipuAI(api_key=ZHIPU_API_KEY)  # 填写您自己的APIKey
with open(r"C:\Users\goldbin\AppData\Local\Temp\gradio\a0addefa04f304b644d9ac4a61cb7b505c41fbdfbb141c09a5635aacb585fa29\audio.wav", "rb") as audio_file:
    resp = client.audio.transcriptions.create(
        model="glm-asr",
        file=audio_file,
        stream=False
    )
    # print(resp)
    print(resp.model_extra['text'])
import io
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI

# 페이지 설정
st.set_page_config(
    page_title="AI 음성 인식 기초 상담",
    page_icon="🎙️",
    layout="centered",
)

# ---------------------------------------------------------
# 1. 왼쪽 사이드바에서 OpenAI API 키 직접 입력
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="화면 왼쪽에 직접 입력한 키는 현재 세션에서만 사용되며 코드에 저장되지 않습니다.",
    )
    st.caption("입력한 키는 브라우저 세션 동안만 유지됩니다.")

st.title("🎙️ AI 음성 인식 기초 상담 앱")
st.caption("마이크로 말씀하신 내용을 AI가 텍스트로 변환하고, 기초 상담을 진행합니다.")

if not api_key or not api_key.strip():
    st.info("왼쪽 사이드바에 **OpenAI API Key**를 입력한 뒤 사용해 주세요.")
    st.stop()

client = OpenAI(api_key=api_key.strip())

# ---------------------------------------------------------
# 2. 세션 상태 초기화 (상담 이력 & 기초 상담 메모)
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "당신은 따뜻하고 공감 능력이 뛰어난 기초 상담 AI입니다. "
                "사용자가 음성으로 전한 고민이나 이야기를 경청하고, 핵심 감정을 정리한 뒤 "
                "부담스럽지 않은 기초 상담 답변과 탐색 질문을 제공하세요. "
                "응답은 한국어로 자연스럽고 차분하게 작성하세요."
            ),
        }
    ]

if "intake_notes" not in st.session_state:
    st.session_state.intake_notes = ""

if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

# ---------------------------------------------------------
# 3. 기초 상담내용 입력 영역
# ---------------------------------------------------------
with st.expander("📋 기초 상담내용 입력 (선택/수정 가능)", expanded=True):
    st.session_state.intake_notes = st.text_area(
        "내담자 기본 정보, 주 호소 문제, 상담 메모 등을 입력하세요.",
        value=st.session_state.intake_notes,
        height=110,
        placeholder="예) 최근 학업/업무 스트레스로 수면 패턴이 불규칙함. 경청과 정서적 지지 위주로 진행.",
    )

st.divider()

# ---------------------------------------------------------
# 4. 마이크 녹음 및 음성 인식(STT) -> 상담 처리
# ---------------------------------------------------------
st.subheader("1. 마이크로 음성 입력")
st.write("아래 마이크 버튼을 눌러 말씀하신 뒤, 다시 눌러 녹음을 완료하세요.")

audio_bytes = audio_recorder(
    text="클릭하여 녹음 시작 / 종료",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    icon_name="microphone",
    icon_size="2x",
    pause_threshold=2.5,
)

# 새 녹음이 들어왔을 때만 STT 및 상담 응답 실행
if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
    st.session_state.last_audio_bytes = audio_bytes
    st.audio(audio_bytes, format="audio/wav")

    try:
        # (A) Whisper API로 음성 -> 텍스트 변환
        with st.spinner("음성을 인식하는 중입니다..."):
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "speech.wav"

            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko",
            )
            recognized_text = transcription.text.strip()

        if not recognized_text:
            st.warning("음성이 명확히 인식되지 않았습니다. 다시 한 번 말씀해 주세요.")
        else:
            st.info(f"**인식된 음성:** {recognized_text}")

            # 기초 상담내용이 있다면 이번 턴의 프롬프트에 맥락으로 결합
            if st.session_state.intake_notes.strip():
                user_prompt = (
                    f"[기초 상담내용]\n{st.session_state.intake_notes.strip()}\n\n"
                    f"[인식된 음성 내용]\n{recognized_text}"
                )
            else:
                user_prompt = recognized_text

            st.session_state.messages.append({"role": "user", "content": user_prompt})

            # (B) Chat Completions API로 기초 상담 응답 생성
            with st.spinner("상담 내용을 정리하고 답변을 생성하는 중입니다..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    temperature=0.7,
                )
                counseling_reply = response.choices[0].message.content.strip()

            st.session_state.messages.append(
                {"role": "assistant", "content": counseling_reply}
            )

    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {e}")

# ---------------------------------------------------------
# 5. 상담 대화 내역 출력 및 초기화
# ---------------------------------------------------------
st.divider()
st.subheader("2. 기초 상담 진행 내역")

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("상담 초기화"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.last_audio_bytes = None
        st.rerun()

# 🏎️ Autonomous AI Car Simulator (PPO Reinforcement Learning)

Este projeto é um ecossistema completo para o desenvolvimento e visualização de um carro autônomo treinado via **Aprendizagem por Reforço (Reinforcement Learning)**. O sistema utiliza inteligência artificial para aprender a navegar em uma cidade desviando de obstáculos e seguindo uma rota de checkpoints.

---

## 🚀 Como o Projeto Funciona

O projeto é dividido em três pilares principais que trabalham em conjunto:

1.  **O Cérebro (IA):** No arquivo `app.py`, utilizamos o algoritmo **PPO (Proximal Policy Optimization)** da biblioteca *Stable Baselines3*. A IA recebe dados de sensores e decide a aceleração e o ângulo do volante.
2.  **O Ambiente (Simulação):** Criamos um ambiente customizado usando **Gymnasium**. Ele simula a física do carro, as colisões com prédios e calcula as recompensas (pontuação) da IA.
3.  **A Visualização (Web):** Através de **WebSockets (SocketIO)**, os dados do Python são enviados em tempo real para o navegador, permitindo assistir ao treino ao vivo.

---

## 📂 Descrição dos Arquivos

### 🧠 1. `app.py` (O Servidor e a IA)
Este é o núcleo do projeto. Ele contém:
- **CityEnv:** A classe que define as regras do mundo (colisões, sensores Lidar e movimentação).
- **Sensores Lidar:** O carro emite 5 raios virtuais para medir a distância de obstáculos.
- **Treinamento:** O modelo aprende por tentativa e erro, buscando maximizar a recompensa ao chegar nos checkpoints.
- **Flask/SocketIO:** Gerencia a comunicação para exibir os gráficos no navegador.

### 🛠️ 2. `editor.html` (O Arquiteto de Mapas)
Uma ferramenta visual para criar novos cenários:
- **Desenhar Prédios:** Clique e arraste para posicionar obstáculos.
- **Checkpoints:** Define o caminho que a IA deve aprender a seguir.
- **Exportação:** Gera automaticamente o código Python das listas `obstacles` e `checkpoints` para você colar no seu `app.py`.

### 📊 3. `index.html` (O Simulador em Tempo Real)
A interface onde você assiste à IA "ganhando vida":
- **Dashboard:** Exibe telemetria como velocidade atual e posição X/Y.
- **Gráficos Canvas:** Renderiza o carro, a estrada dinâmica, os prédios e os feixes dos sensores Lidar conforme a IA processa o ambiente.

---

## 🛠️ Tecnologias Utilizadas

*   **Linguagem:** Python 3.10+
*   **IA/ML:** `stable-baselines3`, `gymnasium`, `numpy`
*   **Servidor Web:** `flask`, `flask-socketio`, `eventlet`
*   **Frontend:** HTML5 Canvas, JavaScript (Socket.io-client)

---

## 🏁 Como Executar o Projeto

1.  **Instale as dependências necessárias:**
    ```bash
    pip install flask flask-socketio eventlet gymnasium stable-baselines3 shimmy numpy
    ```

2.  **Inicie o servidor de treinamento:**
    ```bash
    python app.py
    ```

3.  **Acompanhe o progresso:**
    - Abra o navegador em: `http://localhost:5000`
    - O console do Python mostrará as estatísticas de aprendizado (recompensa média, perda da rede neural, etc).

---

## 📐 Sistema de Recompensas da IA
Para guiar o aprendizado, o algoritmo utiliza:
*   **Avanço na Pista:** Ganha pontos proporcionalmente à redução da distância até o próximo checkpoint.
*   **Checkpoints:** Bônus de **+100** ao passar por um ponto amarelo.
*   **Colisão:** Penalidade de **-200** e reinício imediato do episódio.
*   **Finalização:** Bônus de **+500** ao completar todo o circuito.

---
> *Este projeto foi desenvolvido para fins educacionais em Inteligência Artificial e Robótica Móvel.*

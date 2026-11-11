// RL Bridge - most między TMInterface 2 a Pythonem (tm_env.py / record_line.py).
//
// Instalacja: skopiuj ten plik do  Dokumenty\TMInterface\Plugins\
// Wtyczka nasłuchuje na 127.0.0.1:<rl_port> (domyślnie 8478). Druga instancja gry:
//     set rl_port 8479      a potem      rl_listen
//
// Protokół (little-endian, każda wiadomość zaczyna się od int32 typu):
//   wtyczka -> Python
//     S_STEP        int32 race_time, int32 n, n bajtów SimulationState.ToArray()
//     S_CHECKPOINT  int32 current, int32 target
//     S_MAP_INFO    3 x (int32 n, n bajtów): uid, nazwa, autor (puste, gdy brak mapy)
//     S_INPUTS      int32 n, n bajtów: wejścia bieżącego przejazdu jako skrypt TMInterface
//   Python -> wtyczka
//     C_ACTION      uint8 left, right, up, down        -> ustawia klawisze, gra rusza dalej
//     C_ANALOG      int32 steer, int32 gas             -> wejścia analogowe, gra rusza dalej
//     C_CONTINUE                                       -> gra rusza dalej bez zmiany wejść
//     C_GIVE_UP                                        -> restart wyścigu, gra rusza dalej
//     C_REWIND_START                                   -> powrót do zapisanego startu, odsyła S_STEP
//     C_SAVE_START                                     -> zapamiętuje bieżący stan jako start
//     C_SET_SPEED   float speed
//     C_SET_PERIOD  int32 ms                           -> co ile ms gry wysyłać S_STEP
//     C_EXECUTE     int32 n, n bajtów komendy konsoli
//     C_DISCONNECT                                     -> rozłącza i puszcza grę
//     C_SET_ACTIVE  uint8 on                           -> 0 = nie wysyłaj kroków i puść grę
//     C_GET_MAP                                        -> odsyła S_MAP_INFO
//     C_GET_INPUTS                                     -> odsyła S_INPUTS
//
// Kroki (S_STEP, S_CHECKPOINT) są wysyłane tylko w trybie aktywnym (C_SET_ACTIVE 1).
// Po wysłaniu S_STEP gra STOI, dopóki Python nie odpowie komendą, która ją puszcza
// (ACTION / ANALOG / CONTINUE / GIVE_UP / DISCONNECT).

const string HOST = "127.0.0.1";

enum Msg {
    S_STEP = 1,
    S_CHECKPOINT = 2,
    S_MAP_INFO = 3,
    S_INPUTS = 4,

    C_ACTION = 10,
    C_ANALOG = 11,
    C_CONTINUE = 12,
    C_GIVE_UP = 13,
    C_REWIND_START = 14,
    C_SAVE_START = 15,
    C_SET_SPEED = 16,
    C_SET_PERIOD = 17,
    C_EXECUTE = 18,
    C_DISCONNECT = 19,
    C_SET_ACTIVE = 20,
    C_GET_MAP = 21,
    C_GET_INPUTS = 22
}

Net::Socket@ server = null;
Net::Socket@ client = null;
SimulationState@ startState = null;
int startTime = 0;
int stepPeriod = 50;
bool waiting = false;
bool active = false;
bool analogUsed = false;

PluginInfo@ GetPluginInfo()
{
    auto info = PluginInfo();
    info.Name = "RL Bridge";
    info.Author = "tm-reinforced-learning";
    info.Version = "1.0";
    info.Description = "Sterowanie autem z Pythona (Gymnasium) przez socket TCP";
    return info;
}

void Main()
{
    RegisterVariable("rl_port", 8478.0);
    RegisterVariable("rl_timeout_ms", 600000.0);
    RegisterCustomCommand("rl_listen", "RL Bridge: nasłuchuj ponownie na porcie rl_port", OnListenCommand);
    StartServer();
}

void OnDisabled()
{
    DropClient("wtyczka wyłączona");
    @server = null;
}

void OnListenCommand(int fromTime, int toTime, const string&in commandLine, const array<string>&in args)
{
    StartServer();
}

void StartServer()
{
    DropClient("restart serwera");
    uint16 port = uint16(GetVariableDouble("rl_port"));
    @server = Net::Socket();
    if (!server.Listen(HOST, port)) {
        log("RL Bridge: nie mogę nasłuchiwać na porcie " + port + " (zajęty?)", Severity::Error);
        @server = null;
        return;
    }
    log("RL Bridge: czekam na Pythona na " + HOST + ":" + port);
}

void DropClient(const string&in reason)
{
    if (client !is null) {
        log("RL Bridge: rozłączono (" + reason + ")", Severity::Warning);
        @client = null;
        ClearInputs(GetSimulationManager());
    }
    @startState = null;
    active = false;
}

void ClearInputs(SimulationManager@ sim)
{
    if (sim is null || !sim.InRace)
        return;
    sim.SetInputState(InputType::Left, 0);
    sim.SetInputState(InputType::Right, 0);
    sim.SetInputState(InputType::Up, 0);
    sim.SetInputState(InputType::Down, 0);
    if (analogUsed) {
        sim.SetInputState(InputType::Steer, 0);
        sim.SetInputState(InputType::Gas, 0);
    }
}

// --- komunikacja ------------------------------------------------------------

void SendStep(int raceTime, SimulationState@ state)
{
    array<uint8>@ data = state.ToArray();
    bool ok = client.Write(int(S_STEP))
           && client.Write(raceTime)
           && client.Write(int(data.Length))
           && client.Write(data);
    if (!ok)
        DropClient("błąd wysyłania");
}

bool WriteString(const string&in s)
{
    if (!client.Write(int(s.Length)))
        return false;
    return s.Length == 0 || client.Write(s);
}

void SendMapInfo()
{
    string uid = "";
    string name = "";
    string author = "";
    TM::GameCtnChallenge@ challenge = GetCurrentChallenge();
    if (challenge !is null) {
        uid = challenge.Uid;
        name = challenge.Name;
        author = challenge.Author;
    }
    bool ok = client.Write(int(S_MAP_INFO)) && WriteString(uid) && WriteString(name) && WriteString(author);
    if (!ok)
        DropClient("błąd wysyłania");
}

// Obsługuje jedną komendę. Zwraca true, gdy gra może ruszyć dalej.
bool HandleCommand(SimulationManager@ sim)
{
    int type = client.ReadInt32();
    switch (type) {
        case C_ACTION: {
            int left = client.ReadUint8();
            int right = client.ReadUint8();
            int up = client.ReadUint8();
            int down = client.ReadUint8();
            if (sim.InRace) {
                sim.SetInputState(InputType::Left, left);
                sim.SetInputState(InputType::Right, right);
                sim.SetInputState(InputType::Up, up);
                sim.SetInputState(InputType::Down, down);
            }
            return true;
        }
        case C_ANALOG: {
            int steer = client.ReadInt32();
            int gas = client.ReadInt32();
            analogUsed = true;
            if (sim.InRace) {
                sim.SetInputState(InputType::Steer, steer);
                sim.SetInputState(InputType::Gas, gas);
            }
            return true;
        }
        case C_CONTINUE:
            return true;
        case C_GIVE_UP:
            ClearInputs(sim);
            if (sim.InRace)
                sim.GiveUp();
            return true;
        case C_REWIND_START:
            ClearInputs(sim);
            if (startState !is null && sim.InRace) {
                sim.RewindToState(startState);
                sim.PreventSimulationFinish();
                SendStep(startTime, startState);
            } else {
                log("RL Bridge: brak zapisanego startu - rewind niemożliwy", Severity::Error);
                SendStep(sim.RaceTime, sim.SaveState());
            }
            return false;
        case C_SAVE_START:
            @startState = sim.SaveState();
            startTime = sim.RaceTime;
            sim.PreventSimulationFinish();
            return false;
        case C_SET_SPEED:
            sim.SetSpeed(client.ReadFloat());
            return false;
        case C_SET_PERIOD: {
            int p = client.ReadInt32();
            stepPeriod = p < 10 ? 10 : p;
            return false;
        }
        case C_EXECUTE: {
            int n = client.ReadInt32();
            ExecuteCommand(client.ReadString(n));
            return false;
        }
        case C_DISCONNECT:
            DropClient("Python zakończył");
            return true;
        case C_SET_ACTIVE:
            active = client.ReadUint8() != 0;
            if (!active)
                ClearInputs(sim);   // puść klawisze agenta, żeby nie "trzymały się" po treningu
            return !active;         // wyłączenie trybu aktywnego puszcza czekającą grę
        case C_GET_MAP:
            SendMapInfo();
            return false;
        case C_GET_INPUTS: {
            string inputs = "";
            if (sim.InRace)
                inputs = sim.InputEvents.ToCommandsText();
            if (!(client.Write(int(S_INPUTS)) && WriteString(inputs)))
                DropClient("błąd wysyłania");
            return false;
        }
    }
    log("RL Bridge: nieznana wiadomość " + type, Severity::Error);
    DropClient("błąd protokołu");
    return true;
}

// Blokuje grę, dopóki Python nie każe jechać dalej.
void WaitForCommand(SimulationManager@ sim)
{
    waiting = true;
    uint64 timeout = uint64(GetVariableDouble("rl_timeout_ms"));
    uint64 deadline = Time::Now + timeout;
    while (client !is null) {
        if (client.Available >= 4) {
            if (HandleCommand(sim))
                break;
            deadline = Time::Now + timeout;
        } else if (Time::Now > deadline) {
            DropClient("brak odpowiedzi przez " + timeout + " ms");
            break;
        }
    }
    waiting = false;
}

// --- callbacki TMInterface ---------------------------------------------------

void Render()
{
    if (server is null)
        return;
    if (client is null) {
        Net::Socket@ s = server.Accept(0);
        if (s !is null) {
            @client = s;
            client.NoDelay = true;
            @startState = null;
            active = false;
            analogUsed = false;
            log("RL Bridge: Python połączony", Severity::Success);
        }
        return;
    }
    // komendy wysłane, gdy gra nie czeka (np. ustawienie prędkości zaraz po połączeniu)
    if (!waiting) {
        SimulationManager@ sim = GetSimulationManager();
        while (client !is null && client.Available >= 4)
            HandleCommand(sim);
    }
}

void OnRunStep(SimulationManager@ sim)
{
    if (client is null || waiting || !active)
        return;
    int t = sim.RaceTime;
    if (t % stepPeriod != 0)
        return;
    SendStep(t, sim.SaveState());
    if (client !is null)
        WaitForCommand(sim);
}

void OnCheckpointCountChanged(SimulationManager@ sim, int current, int target)
{
    if (client is null || !active)
        return;
    bool ok = client.Write(int(S_CHECKPOINT)) && client.Write(current) && client.Write(target);
    if (!ok) {
        DropClient("błąd wysyłania");
        return;
    }
    // Meta: oddaj stan od razu - gra mogłaby zakończyć wyścig przed kolejnym krokiem.
    if (current == target && !waiting) {
        SendStep(sim.RaceTime, sim.SaveState());
        if (client !is null)
            WaitForCommand(sim);
    }
}

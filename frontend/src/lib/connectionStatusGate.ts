export interface ConnectionProbeToken {
  requestId: number;
  generation: number;
}

export interface ConnectionStatusGate {
  begin: (generation: number) => ConnectionProbeToken;
  changeContext: (generation: number) => void;
  settle: (token: ConnectionProbeToken) => boolean;
}

export function createConnectionStatusGate(): ConnectionStatusGate {
  let nextRequestId = 1;
  let generation = 0;
  let latestSettledRequestId = 0;

  return {
    begin(nextGeneration) {
      if (nextGeneration !== generation) {
        generation = nextGeneration;
        latestSettledRequestId = 0;
      }
      return { requestId: nextRequestId++, generation };
    },
    changeContext(nextGeneration) {
      generation = nextGeneration;
      latestSettledRequestId = 0;
    },
    settle(token) {
      if (
        token.generation !== generation ||
        token.requestId <= latestSettledRequestId
      ) {
        return false;
      }
      latestSettledRequestId = token.requestId;
      return true;
    }
  };
}

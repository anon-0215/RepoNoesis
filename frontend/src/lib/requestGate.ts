export interface WorkspaceContext {
  generation: number;
  workspaceId: string;
  projectId: string;
  revision: string;
}

export interface RequestToken {
  requestId: number;
  operation: string;
  context: WorkspaceContext;
  previousContext: WorkspaceContext;
}

export interface WorkspaceRequestGate {
  getContext: () => WorkspaceContext;
  beginContextChange: (operation: string, targetWorkspaceId: string) => RequestToken;
  retargetContext: (token: RequestToken, workspaceId: string) => boolean;
  commitContext: (
    token: RequestToken,
    context: Omit<WorkspaceContext, 'generation'>
  ) => boolean;
  restoreContext: (token: RequestToken) => boolean;
  clearContext: (token: RequestToken) => boolean;
  tryEnter: (operation: string) => RequestToken | null;
  isActive: (token: RequestToken) => boolean;
  isCurrent: (token: RequestToken) => boolean;
  finish: (token: RequestToken, requireCurrent?: boolean) => boolean;
}

function sameContext(left: WorkspaceContext, right: WorkspaceContext): boolean {
  return (
    left.generation === right.generation &&
    left.workspaceId === right.workspaceId &&
    left.projectId === right.projectId &&
    left.revision === right.revision
  );
}

export function createRequestGate(): WorkspaceRequestGate {
  let nextRequestId = 1;
  let context: WorkspaceContext = {
    generation: 0,
    workspaceId: '',
    projectId: '',
    revision: ''
  };
  const active = new Map<string, number>();

  function tokenFor(
    operation: string,
    tokenContext: WorkspaceContext,
    previousContext: WorkspaceContext
  ): RequestToken {
    const token = {
      requestId: nextRequestId++,
      operation,
      context: { ...tokenContext },
      previousContext: { ...previousContext }
    };
    active.set(operation, token.requestId);
    return token;
  }

  function isActive(token: RequestToken): boolean {
    return (
      active.get(token.operation) === token.requestId &&
      token.context.generation === context.generation
    );
  }

  return {
    getContext() {
      return { ...context };
    },
    beginContextChange(operation, targetWorkspaceId) {
      const previousContext = { ...context };
      context = {
        generation: context.generation + 1,
        workspaceId: targetWorkspaceId,
        projectId: '',
        revision: ''
      };
      active.clear();
      return tokenFor(operation, context, previousContext);
    },
    retargetContext(token, workspaceId) {
      if (!isActive(token)) return false;
      context = { ...context, workspaceId, projectId: '', revision: '' };
      return true;
    },
    commitContext(token, nextContext) {
      if (!isActive(token)) return false;
      if (context.workspaceId && context.workspaceId !== nextContext.workspaceId) return false;
      context = { generation: context.generation, ...nextContext };
      return true;
    },
    restoreContext(token) {
      if (!isActive(token)) return false;
      context = { ...token.previousContext, generation: context.generation };
      return true;
    },
    clearContext(token) {
      if (!isActive(token)) return false;
      context = {
        generation: context.generation,
        workspaceId: '',
        projectId: '',
        revision: ''
      };
      return true;
    },
    tryEnter(operation) {
      if (active.has(operation)) return null;
      return tokenFor(operation, context, context);
    },
    isActive,
    isCurrent(token) {
      return isActive(token) && sameContext(token.context, context);
    },
    finish(token, requireCurrent = false) {
      if (active.get(token.operation) !== token.requestId) return false;
      active.delete(token.operation);
      return (
        token.context.generation === context.generation &&
        (!requireCurrent || sameContext(token.context, context))
      );
    }
  };
}

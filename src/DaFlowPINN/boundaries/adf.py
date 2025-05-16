import torch

def wall(dim=0, position = 0, direction = 1, order=1):
  if order == 1:
    def f(p):
      d = p[:,dim]-position
      return d * direction
  elif order == 2:
    k = 1000
    def f(p):
      d = p[:,dim]-position
      return d + 1/2 * d**2
      return (d+ 1/(k*2)*(1-torch.exp(-k*(d**2)))) * direction
  return f

def circle(r, center = (0,0), dims = (0,1), inside = False, order = 1, scale=(1,1), shift=(0,0), ratio=1):
    r = r
    center = center
    i = dims[0]
    j = dims[1]

    scale = scale
    shift = shift


    inside = -1 if inside else 1
    if order == 1:
      def f(p):    
          p1 = p[:,i]*scale[0]+shift[0]
          p2 = p[:,j]*scale[1]+shift[1]

          d = d=torch.sqrt((p1-center[0])**2 + ((p2-center[1])*ratio)**2)-r
          return d * inside
    elif order == 2:
      def f(p):    
        d = d=torch.sqrt((p[:,i]-center[0])**2 + (p[:,j]-center[1])**2)-r
        return (d+1/2*(1-torch.exp(-d**2)))*inside

    return f




def line_segment(x0 = (0,0), x1 = (1,1)):
  x0_ = torch.tensor(x0).float()
  x1_ = torch.tensor(x1).float()
  L_ = torch.linalg.norm(x1_-x0_)
  def f(p):
    x0 = x0_.to(p.device)
    x1 = x1_.to(p.device)
    L = L_.to(p.device)
    c = (x0+x1)/2
    d = ((p[:,0]-x0[0])*(x1[1]-x0[1])-(p[:,1]-x0[1])*(x1[0]-x0[0]))/L
    t = 1/L * ((L/2)**2 - torch.linalg.norm(p[:,0:2]-c, dim=1)**2)
    return torch.sqrt(d**2 +((torch.sqrt(t**2+d**4)-t)/2)**2)
  return f


def merge_segments(line_segments, smoothness=1):
    s = smoothness
    line_segments = line_segments
    def f(p):
        merged = 0
        for i in range(len(line_segments)):
          merged += 1/(line_segments[i](p)**s)
        return 1/((merged)**(1/s)) #torch.minimum(sdf_1(p), sdf_2(p)) #
    return f


def intersect(f1, f2, s=0):
  def f(p):
    e = 0.005
    delta = f1(p)-f2(p)
    t = torch.clip((delta+e)/(2*e),0,1)
    beta = 6*t**5 - 15*t**4 + 10*t**3
    return t#(1-beta)*f1(p)+beta*f2(p)
    return (f1(p) + f2(p) - (abs(f1(p))**s + abs(f2(p))**s)**(1/s))
  return f

def unite(f1, f2, order = 1):
  s = order +1
  def f(p):
    return (f1(p) + f2(p) - (abs(f1(p))**s + abs(f2(p))**s)**(1/s))
  return f

def subtract(f1, f2, order = 1):
  s = order + 1
  def f(p):
    return (f1(p) + f2(p) + (abs(f1(p))**s + abs(f2(p))**s)**(1/s))
  return f


# internal:
def _or(f1, f2, order = 1):
  s = order + 1
  return (f1 + f2 + (abs(f1)**s + abs(f2)**s)**(1/s))


def _and(f1, f2, order = 1):
  s = order + 1
  return (f1 + f2 - (abs(f1)**s + abs(f2)**s)**(1/s))

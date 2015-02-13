#include "Win32App.h"


HINSTANCE Win32App::hInstance = 0;
std::string Win32App::sCommandLine;
std::vector<Win32Form*> Win32App::forms_;
Win32AppState Win32App::state_ = Win32AppState::Uninitialized;


void Win32App::RegisterForm(Win32Form* form)
{
  forms_.push_back(form);

  WNDCLASSEX wcex;

  wcex.cbSize         = sizeof(WNDCLASSEX);
  wcex.style          = CS_HREDRAW | CS_VREDRAW;
  wcex.lpfnWndProc    = Win32App::MessageHandler;
  wcex.cbClsExtra     = 0;
  wcex.cbWndExtra     = 0;
  wcex.hInstance      = hInstance;
  wcex.hIcon          = 0;
  wcex.hCursor        = 0;
  wcex.hbrBackground  = 0;
  wcex.lpszMenuName   = 0;
  wcex.lpszClassName  = form->GetFormClassName();
  wcex.hIconSm        = 0;

  int ret = RegisterClassEx(&wcex);
  printf("Regisering %s returned %d\n", form->GetFormClassName(), ret);
}

void Win32App::UnregisterForm(Win32Form* form)
{
  forms_.erase(
    std::remove(forms_.begin(), forms_.end(), form), 
    forms_.end());

  printf("Unregistering form titled: %s\n", form->GetFormClassName());
}

Win32Form* Win32App::FindForm(HWND hWnd)
{
  for(auto pform : forms_)
  {
    if(pform->hWnd == hWnd) return pform;
  }
  return NULL;
}

LRESULT CALLBACK Win32App::MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
  auto pform =  FindForm(hWnd);

  if(pform != NULL) return pform->MessageHandler(message, wParam, lParam);
  else              return DefWindowProc  (hWnd, message, wParam, lParam);

  // int wmId, wmEvent;

  // switch (message)
  // {
  // case WM_COMMAND:
  //   wmId = LOWORD(wParam);
  //   wmEvent = HIWORD(wParam);
  //   // // Parse the menu selections:
  //   // switch (wmId)
  //   // {
  //   //   default:
  //   //     return DefWindowProc(hWnd, message, wParam, lParam);
  //   // }
  //   break;
  // case WM_PAINT:
  //   // this->Paint();
  //   // hdc = BeginPaint(hWnd, &ps);

  //   // EndPaint(hWnd, &ps);
  //   // break;
  // case WM_DESTROY:
  //   PostQuitMessage(0);
  //   break;
  // default:
  //   return DefWindowProc(hWnd, message, wParam, lParam);
  // }    

}

void Win32App::Init()
{
  hInstance = GetModuleHandle(NULL);
  sCommandLine = GetCommandLine();
}

void Win32App::MainLoopInstance()
{
  int ret;
  MSG msg;

  if ((ret = PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) != 0)
  {
    if (ret == -1)
    {
    }
    else //if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
    {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
  }

  for(auto f : forms_)
  {
    f->MainLoopInstance();
  }
}

void Win32App::MainLoop()
{
  while(state_ != Win32AppState::Done)
  {
    MainLoopInstance();

    if(forms_.size() == 0) state_ = Win32AppState::Done;
  }
}

//ATOM RegisterClass()
//{
//  WNDCLASSEX wcex;

//  wcex.cbSize = sizeof(WNDCLASSEX);

//  wcex.style = CS_HREDRAW | CS_VREDRAW;
//  wcex.lpfnWndProc = Win32App::MessageHandler;
//  wcex.cbClsExtra = 0;
//  wcex.cbWndExtra = 0;
//  wcex.hInstance = hInstance;
//  // //wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_RECORDAPP));
//  // //wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
//  // //wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
//  // //wcex.lpszMenuName = MAKEINTRESOURCE(IDC_RECORDAPP);
//  // wcex.lpszClassName = ???;
//  // //wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

//  return RegisterClassEx(&wcex);
//}
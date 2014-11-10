#pragma once




class Win32Dialog
{
private:
public:
  std::string filename;

  Win32Dialog();
  ~Win32Dialog();

  void ShowSave();
  void ShowOpen();
};
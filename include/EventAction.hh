//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file EventAction.hh
/// \brief Definition of the EventAction class

#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

class RunAction;

class EventAction : public G4UserEventAction
{
  public:
    enum EscapeFace
    {
      kEscapeUnknown = 0,
      kEscapeTop = 1,
      kEscapeBottom = 2,
      kEscapeSide = 3
    };

    explicit EventAction(RunAction* runAction);
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event* event) override;
    void EndOfEventAction(const G4Event* event) override;
    void AddInelastic();
    void AddDepositedEnergy(G4double eDep);
    void AddEscapedKineticEnergy(G4double kineticEnergy, G4int particleFlag, G4int escapeFace);

  private:
    void MaybeRotate();
    void RotateFile();
    std::string CurrentFilePath() const;
    G4String BuildFileBaseName() const;

    G4String fBaseName;
    G4int fFileIndex {0};
    G4int fCheckEvery {1000};
    G4long fMaxBytes {1024L * 1024L * 1024L};
    G4int fSplitEveryEvents {0};
    G4bool fInitialized {false};
    RunAction* fRunAction {nullptr};
    G4double fNbInelastic {0.0};
    G4double fPrimaryEnergy {0.0};
    G4bool fHasPrimaryEnergy {false};
    G4double fDepositedEnergy {0.0};
    G4double fEscapedEnergy {0.0};
    G4double fEscapedBackEnergy {0.0};
    G4double fEscapedForwardEnergy {0.0};
    G4double fEscapedLateralEnergy {0.0};
    G4int fEscapedTracks {0};
    G4int fEscapedElectrons {0};
};

#endif

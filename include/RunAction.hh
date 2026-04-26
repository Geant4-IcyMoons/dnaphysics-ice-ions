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
// This example is provided by the Geant4-DNA collaboration
// Any report or published results obtained using the Geant4-DNA software
// shall cite the following Geant4-DNA collaboration publications:
// Med. Phys. 45 (2018) e722-e739
// Phys. Med. 31 (2015) 861-874
// Med. Phys. 37 (2010) 4692-4708
// Int. J. Model. Simul. Sci. Comput. 1 (2010) 157–178
//
// The Geant4-DNA web site is available at http://geant4-dna.org
//
/// \file RunAction.hh
/// \brief Definition of the RunAction class

#ifndef RunAction_h
#define RunAction_h 1

#include "DetectorConstruction.hh"

#include "G4Accumulable.hh"
#include "G4UserRunAction.hh"
#include "globals.hh"

#include <iostream>
#include <string>

class G4Run;

class RunAction : public G4UserRunAction
{
  public:
    enum class LogMode
    {
      kFull,
      kMinimal
    };

    RunAction();
    virtual ~RunAction();

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);
    void AccumulateEventIonisations(G4double nInelastic);
    void AccumulatePrimaryEnergy(G4double energy);

    static void SetLogMode(const G4String& mode);
    static G4String GetLogModeName();
    static G4bool IsFullLogMode();
    static G4bool IsMinimalLogMode();
    static G4bool IsStepModelDetailEnabled();
    static G4bool IsTrackNtupleEnabled();
    static G4bool IsNtupleMergingEnabled();
    G4int GetEventNtupleId() const { return fEventNtupleId; }

  private:
    static LogMode ParseLogMode(const std::string& mode);
    void ConfigureNtuples();

    void PrintWValueSummary(const G4Run* aRun);
    G4int fConfigNtupleId;
    G4int fEventNtupleId;
    G4bool fNtuplesBooked = false;
    G4bool fEnableStringColumns = false;
    LogMode fBookedMode = LogMode::kFull;
    G4Accumulable<G4double> fInelasticSum = 0.0;
    G4Accumulable<G4double> fInelasticSqSum = 0.0;
    G4Accumulable<G4double> fPrimaryEnergySum = 0.0;
    G4Accumulable<G4double> fPrimaryEnergyCount = 0.0;

    static LogMode fLogMode;
    static G4bool fNtupleMergingEnabled;
};
#endif

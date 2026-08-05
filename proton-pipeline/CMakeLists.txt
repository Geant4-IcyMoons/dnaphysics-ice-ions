cmake_minimum_required(VERSION 3.16...3.27)
project(dnaphysics_proton_pipeline)

option(WITH_GEANT4_VIS "Build proton pipeline with Geant4 visualization" ON)
if(WITH_GEANT4_VIS)
  find_package(Geant4 REQUIRED vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

include(${Geant4_USE_FILE})
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/../include
  ${Geant4_INCLUDE_DIR}
)

set(PROTON_SOURCES
  src/ActionInitialization.cc
  src/DetectorConstruction.cc
  src/DetectorMessenger.cc
  src/EventAction.cc
  src/G4DNAEmfietzoglou_iceExcitationStructure.cc
  src/G4DNAEmfietzoglou_iceProtonDcsTable.cc
  src/G4DNAEmfietzoglou_iceProtonExcitationModel.cc
  src/G4DNAEmfietzoglou_iceProtonIonisationModel.cc
  src/G4DNAEmfietzoglou_iceProtonIonisationStructure.cc
  ../src/G4DNANLHHardElastic.cc
  ../src/G4DNANLHHardElasticTable.cc
  src/ModelDataRegistry.cc
  src/PhysicsList_Proton.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/SteppingAction.cc
  src/TrackingAction.cc
)

set(PROTON_HEADERS
  include/ActionInitialization.hh
  include/DetectorConstruction.hh
  include/DetectorMessenger.hh
  include/EventAction.hh
  include/G4DNAEmfietzoglou_iceExcitationStructure.hh
  include/G4DNAEmfietzoglou_iceProtonDcsTable.hh
  include/G4DNAEmfietzoglou_iceProtonExcitationModel.hh
  include/G4DNAEmfietzoglou_iceProtonIonisationModel.hh
  include/G4DNAEmfietzoglou_iceProtonIonisationStructure.hh
  ../include/G4DNANLHHardElastic.hh
  ../include/G4DNANLHHardElasticTable.hh
  include/IcePhaseProperties.hh
  include/ModelDataRegistry.hh
  include/PhysicsList_Proton.hh
  include/PrimaryGeneratorAction.hh
  include/RunAction.hh
  include/SteppingAction.hh
  include/TrackingAction.hh
)

add_executable(dnaphysics_proton dnaphysics_proton.cc ${PROTON_SOURCES} ${PROTON_HEADERS})
target_link_libraries(dnaphysics_proton ${Geant4_LIBRARIES})

set(NLH_CARBON_DATA
  ${PROJECT_SOURCE_DIR}/../python_scripts/physics_ice/nep_mbpol/collision_kernels/geant4/nlh_hard_elastic_C.dat)
set(NLH_CARBON_MANIFEST
  ${PROJECT_SOURCE_DIR}/../python_scripts/physics_ice/nep_mbpol/collision_kernels/geant4/nlh_hard_elastic_C.manifest.json)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/nlh_data)
configure_file(${NLH_CARBON_DATA} ${PROJECT_BINARY_DIR}/nlh_data/ COPYONLY)
configure_file(${NLH_CARBON_MANIFEST} ${PROJECT_BINARY_DIR}/nlh_data/ COPYONLY)
target_compile_definitions(
  dnaphysics_proton PRIVATE DNA_NLH_SOURCE_DATA_FILE="${NLH_CARBON_DATA}")

set(MAC_FILES
  carbon_hard_validation.mac
  e1_proton.mac
  vis.mac
)

foreach(_script ${MAC_FILES})
  configure_file(${_script} ${PROJECT_BINARY_DIR}/. COPYONLY)
endforeach()

install(TARGETS dnaphysics_proton DESTINATION bin)

enable_testing()
add_executable(
  nlh_carbon_table_test
  ../tests/test_nlh_geant4_table.cc
  ../src/G4DNANLHHardElastic.cc
  ../src/G4DNANLHHardElasticTable.cc
  src/ModelDataRegistry.cc
)
target_link_libraries(nlh_carbon_table_test ${Geant4_LIBRARIES})
target_compile_definitions(
  nlh_carbon_table_test PRIVATE DNA_NLH_SOURCE_DATA_FILE="${NLH_CARBON_DATA}")
add_test(
  NAME nlh_carbon_table_test
  COMMAND nlh_carbon_table_test
          ${PROJECT_BINARY_DIR}/nlh_data/nlh_hard_elastic_C.dat
)
